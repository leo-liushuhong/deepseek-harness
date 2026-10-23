/**
 * Incremental chunk-to-message assembler. This is the single canonical assembly
 * algorithm used by the agent loop to build an assistant message from a chunk
 * stream while logging the raw chunks for replay fidelity.
 *
 * @module @deepseek-ai/dsh-llm/assembler
 */

import { brandString } from '@deepseek-ai/dsh-brand'
import { assertNever } from '@deepseek-ai/dsh-util-values'
import type { ToolCallId } from './brand.ts'
import { createAssistantMessage } from './message.ts'
import type { AssistantMessage, ModelMessageSource } from './message.ts'
import type { ContentBlock, FinishReason, ReplayEnvelope, StreamChunk, TokenUsage } from './types.ts'

interface PartialBlock {
  blockType: string
  text: string
  toolCallId?: ToolCallId
  toolCallName?: string
  toolCallArguments: string
  /** Set by `block-end` — authoritative, and freezes the partial. */
  block?: ContentBlock
}

/** Intermediate split of one text block during `<invoke>` recovery. */
type InvokePart =
  | { kind: 'text'; text: string }
  | { kind: 'tool-call'; name: string; parameters: Record<string, string> }

/**
 * Recover tool calls a provider leaked into a `text` block as DeepSeek-native
 * `<invoke name="…">…</invoke>` markup instead of a structured tool-call block.
 * Some gateways serving DeepSeek models over the Anthropic Messages wire emit
 * this markup as plain text; the adapter maps it to a text block and the loop
 * never dispatches the call, retrying until the token limit. Splitting the
 * block into its surrounding text and recovered tool-call blocks lets the call
 * execute. Each recovered `arguments` is re-stringified from the `<parameter>`
 * values, and text before, between, and after the markup is preserved in order.
 * @param blocks - the assembled content blocks, in stream order.
 * @returns the blocks with leaked invokes recovered as tool calls, and whether
 *   any block was split.
 */
function recoverInvokeToolCalls(blocks: ContentBlock[]): { blocks: ContentBlock[]; changed: boolean } {
  const invokePattern = /<invoke name="([^"]+)">([\s\S]*?)<\/invoke>/g
  const parameterPattern = /<parameter name="([^"]+)">([\s\S]*?)<\/parameter>/g
  let changed = false
  let invokeSequence = 0
  const recovered: ContentBlock[] = []
  for (const [index, block] of blocks.entries()) {
    if (block.type !== 'text') {
      recovered.push(block)
      continue
    }
    if (!block.text.includes('<invoke')) {
      recovered.push(block)
      continue
    }
    const parts: InvokePart[] = []
    let cursor = 0
    let matched = false
    for (const match of block.text.matchAll(invokePattern)) {
      const name = match[1]
      const body = match[2]
      if (name === undefined || body === undefined) continue
      matched = true
      if (match.index > cursor) parts.push({ kind: 'text', text: block.text.slice(cursor, match.index) })
      const parameters: Record<string, string> = {}
      for (const parameter of body.matchAll(parameterPattern)) {
        const parameterName = parameter[1]
        const parameterValue = parameter[2]
        if (parameterName === undefined || parameterValue === undefined) continue
        parameters[parameterName] = parameterValue
      }
      parts.push({ kind: 'tool-call', name, parameters })
      cursor = match.index + match[0].length
    }
    if (!matched) {
      recovered.push(block)
      continue
    }
    if (cursor < block.text.length) parts.push({ kind: 'text', text: block.text.slice(cursor) })
    changed = true
    for (const part of parts) {
      if (part.kind === 'text') {
        if (part.text.trim() === '') continue
        recovered.push({ type: 'text', text: part.text })
      } else {
        invokeSequence += 1
        recovered.push({
          type: 'tool-call',
          id: brandString<ToolCallId>(`call-invoke-${index}-${invokeSequence}`),
          name: part.name,
          arguments: JSON.stringify(part.parameters),
        })
      }
    }
  }
  return { blocks: changed ? recovered : blocks, changed }
}

/**
 * Strip DeepSeek-native DSML closing tags that a gateway leaked into a `text`
 * block as literal markup. A closing tag carries no tool name or arguments, so
 * it is pure noise and cannot be recovered into a tool call; removing it keeps
 * the visible answer clean. Covers the fullwidth-pipe form (U+FF5C), the
 * ASCII-pipe form, and the pipe-less form, for the `parameter`, `invoke`, and
 * `tool_calls` tag names.
 * @param blocks - the assembled content blocks, in stream order.
 * @returns the blocks with leaked closing tags removed (a text block that held
 *   only tags is dropped), and whether any block changed.
 */
function stripDsmlClosingTags(blocks: ContentBlock[]): { blocks: ContentBlock[]; changed: boolean } {
  const closingTagPattern = /<\/[\uFF5C|]DSML[\uFF5C|](?:parameter|invoke|tool_calls)>|<\/(?:parameter|invoke|tool_calls)>/g
  let changed = false
  const stripped: ContentBlock[] = []
  for (const block of blocks) {
    if (block.type !== 'text') {
      stripped.push(block)
      continue
    }
    const text = block.text.replace(closingTagPattern, '')
    if (text === block.text) {
      stripped.push(block)
      continue
    }
    changed = true
    if (text.trim() === '') continue
    stripped.push({ type: 'text', text })
  }
  return { blocks: changed ? stripped : blocks, changed }
}

/**
 * Incrementally assembles raw {@link StreamChunk}s into complete
 * {@link ContentBlock}s and a final assistant {@link Message}.
 *
 * The agent loop feeds it while logging raw chunks for replay fidelity, then
 * reads `blocks()` / `message()` / `usage` / `finish` once the stream ends,
 * or `interruptedBlocks()` when cancellation cut the stream short.
 *
 * Tolerant of delta-only protocols (no block-start/end); deltas arriving for
 * an index already closed by `block-end` are ignored (malformed stream) so a
 * misbehaving adapter cannot grow memory or corrupt a completed block.
 */
export class BlockAssembler {
  private partials = new Map<number, PartialBlock>()
  private order: number[] = []
  private _usage: TokenUsage | undefined
  private _finish: FinishReason | undefined
  private _replayState: ReplayEnvelope | undefined

  /**
   * Feed one chunk into the assembly state.
   * @param chunk - the next raw chunk, in stream order.
   */
  push(chunk: StreamChunk): void {
    switch (chunk.type) {
      case 'block-start': {
        if (!this.partials.has(chunk.index)) {
          this.order.push(chunk.index)
          this.partials.set(chunk.index, {
            blockType: chunk.blockType,
            text: '',
            toolCallArguments: '',
          })
        }
        return
      }
      case 'text-delta':
      case 'reasoning-delta': {
        const partial = this.ensure(chunk.index, chunk.type === 'text-delta' ? 'text' : 'reasoning')
        if (partial.block) return // closed by block-end; ignore stragglers
        partial.text += chunk.text
        return
      }
      case 'tool-call-delta': {
        const partial = this.ensure(chunk.index, 'tool-call')
        if (partial.block) return // closed by block-end; ignore stragglers
        partial.toolCallId = chunk.id
        if (chunk.name) partial.toolCallName = chunk.name
        partial.toolCallArguments += chunk.argumentsDelta
        return
      }
      case 'block-end': {
        const partial = this.ensure(chunk.index, chunk.block.type)
        // First close wins; ignoring re-close stragglers keeps streamed output
        // and the final assembled block in agreement.
        if (partial.block) return
        partial.block = chunk.block
        return
      }
      case 'usage': {
        this._usage = chunk.usage
        return
      }
      case 'finish': {
        this._finish = chunk.reason
        this._replayState = chunk.replayState
        return
      }
      default: return assertNever(chunk, 'BlockAssembler.push')
    }
  }

  private ensure(index: number, blockType: string): PartialBlock {
    let partial = this.partials.get(index)
    if (!partial) {
      partial = { blockType, text: '', toolCallArguments: '' }
      this.partials.set(index, partial)
      this.order.push(index)
    }
    return partial
  }

  private assemble(partial: PartialBlock, index: number): ContentBlock {
    if (partial.block) return partial.block
    switch (partial.blockType) {
      case 'text': return { type: 'text', text: partial.text }
      case 'reasoning': return { type: 'reasoning', text: partial.text }
      case 'tool-call': return {
        type: 'tool-call',
        id: partial.toolCallId ?? brandString<ToolCallId>(`call-${index}`),
        name: partial.toolCallName ?? '',
        arguments: partial.toolCallArguments,
      }
      default: throw new Error(`cannot assemble incomplete block of type "${partial.blockType}"`)
    }
  }

  /** Invariant accessor: every index in `order` has a partial. */
  private mustGet(index: number): PartialBlock {
    const partial = this.partials.get(index)
    if (!partial) throw new Error(`BlockAssembler invariant violated: no partial for index ${index}`)
    return partial
  }

  /**
   * The one shared keep/drop decision over all seen blocks: max-token
   * truncation drops tool calls that cannot be executed safely. Emitted blocks
   * and replay metadata both derive from this result, so they cannot disagree.
   */
  private assembled(): { blocks: ContentBlock[]; replay: ReplayEnvelope | undefined } {
    const all = this.order.map(index => this.assemble(this.mustGet(index), index))
    const recovered = recoverInvokeToolCalls(all)
    const stripped = stripDsmlClosingTags(recovered.blocks)
    const base = stripped.blocks
    const changed = recovered.changed || stripped.changed
    const kept = this.finish.kind === 'max-tokens'
      ? base.map(block => block.type !== 'tool-call')
      : undefined
    const blocks = kept === undefined ? base : base.filter((_, position) => kept[position])
    const envelope = this._replayState
    if (envelope?.blocks === undefined) return { blocks, replay: envelope }
    // Recovery or stripping rewrites text blocks, so the adapter's per-block
    // replay metadata no longer aligns with the emitted blocks; drop it rather
    // than fabricate adapter-private metadata this provider-neutral assembler
    // cannot know.
    if (changed || envelope.blocks.length !== all.length) return { blocks, replay: undefined }
    return {
      blocks,
      replay: kept === undefined || blocks.length === all.length
        ? envelope
        : { response: envelope.response, blocks: envelope.blocks.filter((_, position) => kept[position]) },
    }
  }

  /**
   * Assemble all blocks seen so far, in stream order.
   * @returns one block per seen index, except that max-token truncation drops
   *   tool calls that cannot be executed safely; an open block assembles from
   *   its accumulated deltas (an unknown block type never closed by `block-end` throws).
   */
  blocks(): ContentBlock[] {
    return this.assembled().blocks
  }

  /**
   * Assemble the prefix an interrupted stream can safely finalize: closed and
   * open text/reasoning blocks with non-whitespace content, in stream order.
   * Tool calls are omitted because interruption precedes dispatch; retaining
   * one would require a fabricated result. Open unknown blocks are also omitted.
   * @returns the kept blocks; empty when nothing streamed before the interruption.
   */
  interruptedBlocks(): ContentBlock[] {
    return this.order
      .map((index) => {
        const partial = this.mustGet(index)
        const type = partial.block?.type ?? partial.blockType
        if (type !== 'text' && type !== 'reasoning') return undefined
        return this.assemble(partial, index)
      })
      .filter((block): block is ContentBlock =>
        (block?.type === 'text' || block?.type === 'reasoning') && block.text.trim() !== '')
  }

  /** Usage from the `usage` chunk; undefined until one arrives. */
  get usage(): TokenUsage | undefined {
    return this._usage
  }

  /** Finish reason from the `finish` chunk; `{kind: 'stop'}` when the stream ended without one. */
  get finish(): FinishReason {
    return this._finish ?? { kind: 'stop' }
  }

  /**
   * Replay metadata from the terminal finish chunk, if any, with per-block
   * entries pruned in step with {@link blocks}. Undefined when the envelope's
   * entries do not align with the emitted blocks.
   */
  get replayState(): ReplayEnvelope | undefined {
    return this.assembled().replay
  }

  /**
   * The assembled assistant message.
   * @param source - provider/model attribution (without the `kind` tag) for the assembled message.
   * @returns a frozen assistant-role message over `blocks()` (same open-block assembly rules).
   */
  message(source: Omit<ModelMessageSource, 'kind'>): AssistantMessage {
    return createAssistantMessage({ content: this.blocks(), source })
  }
}
