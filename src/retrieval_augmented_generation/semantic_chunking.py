import re
import pandas as pd
from src.common.config import RagConfig, config



class SemanticChunkingService:

    def __init__(self, rag_config: RagConfig | None = None) -> None:
        self.config = rag_config or config.rag



    def count_tokens_approx(self, text: str) -> int:
        if not isinstance(text, str) or not text.strip():
            return 0

        return len(re.findall(r"\w+|[^\w\s]", text))



    def split_paragraphs(
        self,
        text: str,
        sentence_per_newline_threshold: float = 1.55,
    ) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        raw_text = text.strip()
        non_empty_lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        if not non_empty_lines:
            return []

        total_newline_blocks = len(non_empty_lines)

        normalized_text = re.sub(r"\s+", " ", raw_text)
        total_sentences = len(self.split_sentences_in_paragraph(normalized_text))

        sentence_per_line_ratio = (
            total_sentences / total_newline_blocks
            if total_newline_blocks > 0
            else total_sentences
        )

        sentence_per_line_format = sentence_per_line_ratio <= sentence_per_newline_threshold

        if sentence_per_line_format:
            merged_text = re.sub(r"\s+", " ", raw_text).strip()
            return [merged_text]

        return [
            re.sub(r"\s+", " ", line).strip()
            for line in non_empty_lines
        ]



    def split_sentences_in_paragraph(self, paragraph: str) -> list[str]:
        if not isinstance(paragraph, str) or not paragraph.strip():
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9\"'])",
            paragraph.strip(),
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]



    def sentence_starts_with_transition(self, sentence: str) -> bool:
        transition_patterns = [
            # contrast / opposition
            "however",
            "but",
            "yet",
            "still",
            "nevertheless",
            "nonetheless",
            "even so",
            "even then",
            "despite this",
            "despite that",
            "in contrast",
            "by contrast",
            "on the other hand",
            "on the contrary",
            "conversely",
            "instead",
            "rather",
            "alternatively",

            # continuation / addition
            "meanwhile",
            "also",
            "furthermore",
            "moreover",
            "in addition",
            "additionally",
            "besides",
            "what is more",
            "more broadly",
            "more importantly",
            "at the same time",
            "similarly",
            "likewise",

            # cause / consequence
            "therefore",
            "thus",
            "as a result",
            "consequently",
            "accordingly",
            "for this reason",
            "because of this",
            "because of that",
            "this means",
            "that means",
            "this suggests",
            "this indicates",
            "this reflects",
            "this underscores",
            "this highlights",
            "this marks",
            "this signals",

            # emphasis / clarification
            "indeed",
            "in fact",
            "notably",
            "significantly",
            "importantly",
            "crucially",
            "to be clear",
            "in other words",
            "put differently",
            "in practical terms",
            "in this context",
            "under these conditions",

            # examples / evidence
            "for example",
            "for instance",
            "such as",
            "including",
            "according to",
            "citing",
            "as evidence",
            "for comparison",

            # time / sequence
            "then",
            "next",
            "later",
            "earlier",
            "previously",
            "subsequently",
            "afterward",
            "afterwards",
            "since then",
            "from there",
            "over time",
            "in recent years",
            "in recent months",
            "in recent weeks",
            "in the meantime",

            # topic shift / section movement
            "separately",
            "elsewhere",
            "in another development",
            "on another front",
            "turning to",
            "regarding",
            "as for",
            "with respect to",
            "when it comes to",
            "in terms of",

            # conclusion / summary
            "overall",
            "ultimately",
            "finally",
            "in conclusion",
            "to conclude",
            "in summary",
            "to summarize",
            "the result is",
            "the outcome is",
        ]

        sentence_lower = sentence.lower().strip()

        return any(
            sentence_lower.startswith(pattern)
            for pattern in transition_patterns
        )



    def build_sentence_units(self, text: str) -> list[dict]:
        units = []

        paragraphs = self.split_paragraphs(text)

        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = self.split_sentences_in_paragraph(paragraph)

            for sentence_index, sentence in enumerate(sentences):
                units.append({
                    "text": sentence,
                    "tokens": self.count_tokens_approx(sentence),
                    "paragraph_index": paragraph_index,
                    "sentence_index": sentence_index,
                    "is_paragraph_end": sentence_index == len(sentences) - 1,
                    "starts_with_transition": self.sentence_starts_with_transition(sentence),
                })

        return units



    def get_overlap_units(self, previous_chunk_units: list[dict], overlap_tokens: int) -> list[dict]:
        if not previous_chunk_units:
            return []

        overlap_units = []
        token_count = 0

        for unit in reversed(previous_chunk_units):
            overlap_units.insert(0, unit)
            token_count += unit["tokens"]

            if token_count >= overlap_tokens:
                break

        return overlap_units



    def expand_final_chunk_with_previous_context(
        self,
        current_units: list[dict],
        previous_chunk_units: list[dict],
        min_chunk_tokens: int,
    ) -> list[dict]:
        current_tokens = sum(unit["tokens"] for unit in current_units)
        
        if current_tokens >= min_chunk_tokens:
            return current_units

        current_texts = {unit["text"] for unit in current_units}
        expanded_units = current_units.copy()

        for unit in reversed(previous_chunk_units):
            if unit["text"] in current_texts:
                continue

            expanded_units.insert(0, unit)
            current_texts.add(unit["text"])
            current_tokens += unit["tokens"]

            if current_tokens >= min_chunk_tokens:
                break

        return expanded_units



    def chunk_text_semantic_sentence_aware(
        self,
        text: str,
        min_chunk_tokens: int | None = None,
        max_chunk_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> list[dict]:
        
        min_chunk_tokens = min_chunk_tokens or self.config.chunk_min_tokens
        max_chunk_tokens = max_chunk_tokens or self.config.chunk_max_tokens
        overlap_tokens = overlap_tokens or self.config.chunk_overlap_tokens

        units = self.build_sentence_units(text)

        if not units:
            return []

        chunks = []
        previous_chunk_units = []
        i = 0

        while i < len(units):
            
            current_units = self.get_overlap_units(
                previous_chunk_units,
                overlap_tokens,
            )

            current_tokens = sum(unit["tokens"] for unit in current_units)
            added_new_sentence = False
            stop_reason = "end_of_document"

            while i < len(units):
                
                next_unit = units[i]

                if added_new_sentence and current_tokens + next_unit["tokens"] > max_chunk_tokens:
                    stop_reason = "max_token_stop"
                    break

                current_units.append(next_unit)
                current_tokens += next_unit["tokens"]
                added_new_sentence = True
                i += 1

                next_sentence = units[i] if i < len(units) else None

                reached_min_size = current_tokens >= min_chunk_tokens
                at_paragraph_boundary = next_unit["is_paragraph_end"]
                
                next_starts_transition = (
                    next_sentence is not None
                    and next_sentence["starts_with_transition"]
                )
                
                next_would_exceed_max = (
                    next_sentence is not None
                    and current_tokens + next_sentence["tokens"] > max_chunk_tokens
                )

                if i >= len(units):
                    stop_reason = "end_of_document"
                    break

                if reached_min_size and at_paragraph_boundary:
                    stop_reason = "paragraph_boundary"
                    break

                if reached_min_size and next_starts_transition:
                    stop_reason = "transition_word"
                    break

                if next_would_exceed_max:
                    stop_reason = "max_token_stop"
                    break

            if i >= len(units):
                current_units = self.expand_final_chunk_with_previous_context(
                    current_units=current_units,
                    previous_chunk_units=previous_chunk_units,
                    min_chunk_tokens=min_chunk_tokens,
                )

                stop_reason = "end_of_document"

            chunk_text = " ".join(unit["text"] for unit in current_units).strip()

            if chunk_text:
                chunks.append({
                    "chunk_text": chunk_text,
                    "chunk_token_count": self.count_tokens_approx(chunk_text),
                    "stop_reason": stop_reason,
                })

            previous_chunk_units = current_units

        return chunks



    def chunk_news_for_embedding(
        self,
        df: pd.DataFrame,
        uuid_col: str = "uuid",
        text_col: str = "full_text",
        min_chunk_tokens: int | None = None,
        max_chunk_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> pd.DataFrame:
        rows = []

        for _, row in df.iterrows():
            uuid = row[uuid_col]
            text = row[text_col]

            if pd.isna(text) or not str(text).strip():
                continue

            chunks = self.chunk_text_semantic_sentence_aware(
                text=str(text),
                min_chunk_tokens=min_chunk_tokens,
                max_chunk_tokens=max_chunk_tokens,
                overlap_tokens=overlap_tokens,
            )

            for chunk_index, chunk in enumerate(chunks, start=1):
                rows.append({
                    "uuid": uuid,
                    "chunk_id": chunk_index,
                    "chunk_text": chunk["chunk_text"],
                    "chunk_token_count": chunk["chunk_token_count"],
                    "stop_reason": chunk["stop_reason"],
                })

        return pd.DataFrame(
            rows,
            columns=[
                "uuid",
                "chunk_id",
                "chunk_text",
                "chunk_token_count",
                "stop_reason",
            ],
        )

