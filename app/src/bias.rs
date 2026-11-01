//! Phrases learned from corrections, and the token-level bonus they add to the
//! TDT joint network.
//!
//! A correction is reduced to the character runs that differ between the
//! recognized text and what the user typed. Each run is encoded with the
//! model's SentencePiece vocabulary, so that it becomes a token sequence the
//! decoder can watch for. While decoding, whenever the emitted tokens end in a
//! phrase prefix, the phrase's next token gets its weight added to the joint
//! logit: corrected wording is favoured, recognized wording that the user
//! removed is pushed down.
//!
//! The text around a correction is kept on both sides. The left side is a set
//! of context tiers: the phrase pulls at a quarter strength everywhere, half
//! with the last half of its context, and fully with all of it. The right side
//! cannot be seen while a left-to-right decoder chooses the phrase, so it is
//! scored on the finished hypotheses instead, where a path that continued the
//! way the user did outranks one that did not.
//!
//! The merge order is SentencePiece's: join the adjacent pair whose combined
//! piece has the lowest vocabulary index first. That is the same order the
//! model was trained with, which is what makes the stored phrases line up with
//! the tokens the decoder emits.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::config::Config;

/// A phrase to favour (positive weight) or avoid (negative weight).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phrase {
    /// What the user typed. A deletion has an empty one.
    #[serde(alias = "text")]
    pub to: String,
    /// What was recognized where the user put `to`. Kept with it so that a
    /// correction is one rule: the replacement is favoured and the wording it
    /// replaced is pushed down by the same rule, in the same context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(default = "unit_weight")]
    pub weight: f32,
    /// Text that came immediately before the phrase when it was learned. The
    /// phrase pulls a little with no match, more with half of it, and at full
    /// strength with the whole of it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<String>,
    /// Text that followed the phrase when it was learned. A left-to-right
    /// decoder cannot see it while choosing the phrase, so it ranks finished
    /// hypotheses instead: one that continues the same way wins the tie.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
}

fn unit_weight() -> f32 {
    1.0
}

/// The `bias.yaml` document. Editable by hand as well as by [`learn`].
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BiasFile {
    #[serde(default)]
    pub phrases: Vec<Phrase>,
}

impl BiasFile {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_yaml::to_string(self).context("writing bias.yaml")?;
        let tmp = path.with_extension("yaml.tmp");
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Add `delta` to a phrase, creating it when it is new. Extremes are
    /// clamped so that one repeated correction cannot drown out the acoustic
    /// model.
    pub fn add(
        &mut self,
        to: &str,
        delta: f32,
        from: Option<&str>,
        before: Option<&str>,
        after: Option<&str>,
    ) {
        let to = to.trim();
        let from = from.map(str::trim).filter(|f| !f.is_empty());
        if to.is_empty() && from.is_none() {
            return;
        }
        let before = before.map(str::trim).filter(|b| !b.is_empty());
        let after = after.map(str::trim).filter(|a| !a.is_empty());
        if let Some(existing) = self.phrases.iter_mut().find(|p| {
            p.to == to
                && p.from.as_deref() == from
                && p.before.as_deref() == before
                && p.after.as_deref() == after
        }) {
            existing.weight = (existing.weight + delta).clamp(-5.0, 5.0);
        } else {
            self.phrases.push(Phrase {
                to: to.to_string(),
                from: from.map(str::to_string),
                weight: delta.clamp(-5.0, 5.0),
                before: before.map(str::to_string),
                after: after.map(str::to_string),
            });
        }
    }

    /// Drop every rule for `text`, on either side of the replacement. Returns
    /// whether anything was there.
    pub fn remove(&mut self, text: &str) -> bool {
        let text = text.trim();
        let before = self.phrases.len();
        self.phrases
            .retain(|p| p.to != text && p.from.as_deref() != Some(text));
        before != self.phrases.len()
    }
}

/// One piece while it is still being merged, with the characters it covers.
struct Symbol {
    text: String,
    start: usize,
    end: usize,
}

/// SentencePiece BPE over the exported vocabulary.
pub struct Tokenizer {
    index: HashMap<String, usize>,
    unk: usize,
}

impl Tokenizer {
    pub fn new(vocab: &[String]) -> Self {
        let index = vocab.iter().enumerate().map(|(i, p)| (p.clone(), i)).collect();
        let unk = vocab.iter().position(|p| p == "<unk>").unwrap_or(0);
        Self { index, unk }
    }

    /// Encode `text` the way the model's tokenizer does. The vocabulary holds
    /// every Japanese character these phrases are made of; anything missing
    /// falls back to `<unk>` and the phrase is then skipped by [`CompiledBias`].
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.encode_with_prefix(text, true)
    }

    /// Encode text that continues an utterance: a context, a phrase, or the
    /// text after one. The leading `▁` that [`encode`](Tokenizer::encode) adds
    /// is only emitted at a word start, so a phrase encoded with it could
    /// never match the middle of a sentence, which is where most corrections
    /// land.
    pub fn encode_context(&self, text: &str) -> Vec<usize> {
        self.encode_with_prefix(text, false)
    }

    /// Encode `text` and give each token the range of characters in `text` it
    /// came from. Corrections use the ranges to cut a context and a phrase on
    /// token boundaries: a context that starts in the middle of a token
    /// re-encodes to other tokens and would never match what the decoder
    /// emitted.
    pub fn encode_spans(&self, text: &str) -> Vec<(usize, usize, usize)> {
        let mut symbols = self.symbols(text, false);
        self.merge(&mut symbols);
        let mut tokens = Vec::new();
        for symbol in &symbols {
            if let Some(&id) = self.index.get(&symbol.text) {
                tokens.push((id, symbol.start, symbol.end));
                continue;
            }
            for byte in symbol.text.as_bytes() {
                let piece = format!("<0x{byte:02X}>");
                let id = self.index.get(&piece).copied().unwrap_or(self.unk);
                tokens.push((id, symbol.start, symbol.end));
            }
        }
        tokens
    }

    fn symbols(&self, text: &str, word_start: bool) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        if word_start {
            symbols.push(Symbol { text: "\u{2581}".into(), start: 0, end: 0 });
        }
        for (i, c) in text.chars().enumerate() {
            // NFKC per character keeps the original index; a character that
            // expands becomes one symbol holding all of it so that its range
            // stays whole.
            let mut normalized = String::new();
            for n in c.to_string().nfkc() {
                normalized.push(if n == ' ' { '\u{2581}' } else { n });
            }
            if normalized.is_empty() {
                continue;
            }
            symbols.push(Symbol { text: normalized, start: i, end: i + 1 });
        }
        symbols
    }

    fn merge(&self, symbols: &mut Vec<Symbol>) {
        loop {
            let mut best: Option<(usize, usize)> = None; // (piece index, position)
            for i in 0..symbols.len().saturating_sub(1) {
                let merged = format!("{}{}", symbols[i].text, symbols[i + 1].text);
                if let Some(&piece) = self.index.get(&merged) {
                    if best.map_or(true, |(b, _)| piece < b) {
                        best = Some((piece, i));
                    }
                }
            }
            let Some((_, pos)) = best else { break };
            let right = symbols.remove(pos + 1);
            let left = &mut symbols[pos];
            left.text.push_str(&right.text);
            left.end = right.end;
        }
    }

    fn encode_with_prefix(&self, text: &str, word_start: bool) -> Vec<usize> {
        let mut symbols = self.symbols(text, word_start);
        self.merge(&mut symbols);

        // A symbol that never merged is looked up as is. If the vocabulary has
        // no such piece, SentencePiece falls back to one `<0xXX>` piece per
        // UTF-8 byte; only a byte with no piece either becomes <unk>.
        let mut ids = Vec::new();
        for symbol in &symbols {
            if let Some(&id) = self.index.get(&symbol.text) {
                ids.push(id);
                continue;
            }
            for byte in symbol.text.as_bytes() {
                let piece = format!("<0x{byte:02X}>");
                ids.push(self.index.get(&piece).copied().unwrap_or(self.unk));
            }
        }
        ids
    }

    pub fn unk(&self) -> usize {
        self.unk
    }
}

/// Weight of a phrase's first token, relative to its full weight. The rest of
/// the phrase ramps up to 1.0 as more of the prefix is emitted.
const PREFIX_FLOOR: f32 = 0.3;

/// Shares of a phrase's weight for none, half and all of its preceding
/// context. They add up to one, so a phrase that meets its context exactly is
/// as strong as before, one whose last half matches still pulls, and one whose
/// context is different pulls a little. A phrase learned without context has
/// no tiers: it pulls everywhere at full weight.
const BASE_SHARE: f32 = 0.25;
const HALF_SHARE: f32 = 0.25;
const FULL_SHARE: f32 = 0.5;

/// Share of a phrase's weight that its learned continuation adds to a finished
/// hypothesis.
const RIGHT_SHARE: f32 = 0.5;

/// How much of a rule's weight the wording it replaced is pushed down by.
/// Suppression is weaker than boosting: a wrong penalty can hide a word the
/// microphone actually heard.
const SUPPRESS: f32 = 0.6;

/// How many different surroundings a Japanese spelling of a Latin word must be
/// corrected in before the correction is taken to be about the word and not
/// about where it stood in the sentence.
const WORD_CONTEXTS: usize = 3;

/// Phrases compiled into prefix lookups for the decoder.
pub struct CompiledBias {
    /// Emitted-token prefix -> the tokens that may follow it, with weights.
    table: HashMap<Vec<usize>, Vec<(usize, f32)>>,
    max_prefix: usize,
    /// Phrases together with the context on both sides of them, scored on
    /// finished hypotheses rather than while emitting.
    right: Vec<(Vec<usize>, f32)>,
    /// Replacements applied to the finished text. Some corrections cannot be
    /// reached through the logits at all: an uppercase Latin letter is a byte
    /// piece that the language prior gives no chance against the lowercase
    /// one, and a Latin word is a reading the prior prefers to spell in
    /// Japanese, so no weight the decoder can take would flip either. The
    /// correction is applied to the text instead.
    text: Vec<TextRule>,
}

/// A replacement the finished text goes through. `anywhere` marks a rule that
/// is about the word rather than about a place in a sentence: a case change,
/// or a Latin word the recognizer writes in Japanese and the user corrected in
/// several different sentences. Every other rule needs its recorded
/// surroundings.
struct TextRule {
    from: String,
    to: String,
    before: Option<String>,
    after: Option<String>,
    anywhere: bool,
}

impl TextRule {
    fn fires(&self, before_text: &str, after_text: &str) -> bool {
        let before = match &self.before {
            Some(before) => before_text.trim_end().ends_with(before.as_str()),
            None => true,
        };
        let after = match &self.after {
            Some(after) => after_text.trim_start().starts_with(after.as_str()),
            None => true,
        };
        before && after
    }
}

impl CompiledBias {
    pub fn empty() -> Self {
        Self { table: HashMap::new(), max_prefix: 0, right: Vec::new(), text: Vec::new() }
    }

    /// Apply the learned replacements to a transcript. Longest first, one pass,
    /// so a rule cannot undo another by matching its own output.
    pub fn rewrite(&self, text: &str) -> String {
        if self.text.is_empty() {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len());
        let mut pos = 0;
        while pos < text.len() {
            let rest = &text[pos..];
            let mut applied = None;
            for rule in &self.text {
                let Some(after) = rest.strip_prefix(rule.from.as_str()) else {
                    continue;
                };
                let fires = if rule.anywhere {
                    let before = text[..pos].chars().next_back();
                    let after_char = after.chars().next();
                    let bounded = |c: Option<char>| {
                        c.map_or(true, |c| !c.is_ascii_alphanumeric())
                    };
                    bounded(before) && bounded(after_char)
                } else {
                    rule.fires(&text[..pos], after)
                };
                if fires {
                    applied = Some((rule, after));
                    break;
                }
            }
            match applied {
                Some((rule, after)) => {
                    out.push_str(&rule.to);
                    pos = text.len() - after.len();
                }
                None => {
                    let c = rest.chars().next().unwrap();
                    out.push(c);
                    pos += c.len_utf8();
                }
            }
        }
        out
    }

    /// `scale` multiplies every stored weight, so the strength can be tuned
    /// without rewriting `bias.yaml`.
    pub fn compile(phrases: &[Phrase], tokenizer: &Tokenizer, scale: f32) -> Self {
        let mut table: HashMap<Vec<usize>, Vec<(usize, f32)>> = HashMap::new();
        let mut right = Vec::new();
        let mut text_rules = Vec::new();
        let mut max_prefix = 0usize;
        let words = word_rules(phrases);
        for phrase in phrases {
            if phrase.weight == 0.0 {
                continue;
            }
            let to_text = phrase.to.trim();
            let from_text = phrase.from.as_deref().map(str::trim).unwrap_or("");
            if phrase.weight > 0.0
                && !from_text.is_empty()
                && !to_text.is_empty()
                && from_text != to_text
            {
                text_rules.push(TextRule {
                    from: from_text.to_string(),
                    to: to_text.to_string(),
                    before: phrase.before.clone(),
                    after: phrase.after.clone(),
                    anywhere: from_text.to_lowercase() == to_text.to_lowercase()
                        || words.contains(&(from_text.to_string(), to_text.to_string())),
                });
            }
            let context = match phrase.before.as_deref() {
                Some(before) => {
                    let encoded = tokenizer.encode_context(before);
                    if encoded.iter().any(|&t| t == tokenizer.unk()) {
                        Vec::new()
                    } else {
                        encoded
                    }
                }
                None => Vec::new(),
            };
            let to = tokenizer.encode_context(&phrase.to);
            let from = phrase
                .from
                .as_deref()
                .map(|from| tokenizer.encode_context(from))
                .unwrap_or_default();
            if to.contains(&tokenizer.unk()) || from.contains(&tokenizer.unk()) {
                continue;
            }

            // The replacement is one rule with two sides: the wording the user
            // chose pulls, the wording it replaced is pushed down. They share
            // the context, so they meet at the same place in the utterance.
            let keep = phrase.weight * scale;
            let drop = -SUPPRESS * keep;
            max_prefix = max_prefix.max(enter(&mut table, &context, &to, keep));
            max_prefix = max_prefix.max(enter(&mut table, &context, &from, drop));

            // What the user went on to say. Only wording the user chose pulls,
            // and only after it and its whole context: a hypothesis that
            // continues the same way outranks one that does not.
            if phrase.weight > 0.0 && !to.is_empty() {
                if let Some(after) = phrase.after.as_deref() {
                    let encoded = tokenizer.encode_context(after);
                    if !encoded.is_empty() && !encoded.iter().any(|&t| t == tokenizer.unk()) {
                        let mut pattern = context.clone();
                        pattern.extend_from_slice(&to);
                        pattern.extend_from_slice(&encoded);
                        right.push((pattern, phrase.weight * scale * RIGHT_SHARE));
                    }
                }
            }
        }
        // Longest first, so `LB LT` wins over `LB` at the same position.
        text_rules.sort_by(|a, b| b.from.chars().count().cmp(&a.from.chars().count()));
        Self { table, max_prefix, right, text: text_rules }
    }

    /// The score a finished token sequence gets for continuing the way the
    /// user did after a learned phrase. One hit per phrase, so a word repeated
    /// in an utterance cannot pile up.
    pub fn right_bonus(&self, emitted: &[usize]) -> f32 {
        let mut total = 0.0;
        for (pattern, weight) in &self.right {
            if pattern.len() > emitted.len() {
                continue;
            }
            if emitted.windows(pattern.len()).any(|w| w == pattern.as_slice()) {
                total += weight;
            }
        }
        total
    }

    /// Fill `bonus` with the weight each label carries given the tokens emitted
    /// so far. `bonus` must have one slot per label.
    pub fn fill(&self, emitted: &[usize], bonus: &mut [f32]) {
        bonus.fill(0.0);
        if self.table.is_empty() {
            return;
        }
        // Length 0 is the empty prefix: a phrase can begin after any emitted
        // token, including at the start of the utterance.
        let max = self.max_prefix.min(emitted.len());
        for len in 0..=max {
            let prefix = &emitted[emitted.len() - len..];
            if let Some(next) = self.table.get(prefix) {
                for &(token, weight) in next {
                    if let Some(slot) = bonus.get_mut(token) {
                        *slot += weight;
                    }
                }
            }
        }
    }
}

/// The Japanese-to-Latin replacements seen in enough different surroundings to
/// be about the word rather than about the sentence it stood in. The decoder
/// cannot be talked into a Latin spelling over the kanji the language prior
/// prefers, so these corrections are made on the finished text, and a rule
/// that only fits one sentence would never fire on the next one.
fn word_rules(phrases: &[Phrase]) -> HashSet<(String, String)> {
    let mut seen: HashMap<(String, String), HashSet<(String, String)>> = HashMap::new();
    for phrase in phrases {
        if phrase.weight <= 0.0 {
            continue;
        }
        let to = phrase.to.trim();
        let Some(from) = phrase.from.as_deref().map(str::trim) else {
            continue;
        };
        if from.is_empty() || from == to || !to.chars().all(|c| c.is_ascii()) {
            continue;
        }
        if from.chars().all(|c| c.is_ascii()) {
            continue;
        }
        seen.entry((from.to_string(), to.to_string())).or_default().insert((
            phrase.before.clone().unwrap_or_default(),
            phrase.after.clone().unwrap_or_default(),
        ));
    }
    seen.into_iter()
        .filter(|(_, contexts)| contexts.len() >= WORD_CONTEXTS)
        .map(|(pair, _)| pair)
        .collect()
}

/// Enter one side of a rule at every context tier, returning the length of the
/// longest key added.
fn enter(
    table: &mut HashMap<Vec<usize>, Vec<(usize, f32)>>,
    context: &[usize],
    tokens: &[usize],
    weight: f32,
) -> usize {
    if tokens.is_empty() || weight == 0.0 {
        return 0;
    }
    let context_len = context.len();
    let mut full = context.to_vec();
    full.extend_from_slice(tokens);

    // The same side is entered once per context tier. The keys are suffixes of
    // the context followed by the phrase prefix, so the ordinary suffix match
    // in `fill` picks whichever tier the emitted text reaches, and a full
    // match sums to the whole weight.
    let tiers: Vec<(usize, f32)> = match context_len {
        0 => vec![(0, 1.0)],
        1 => vec![(0, BASE_SHARE), (1, FULL_SHARE + HALF_SHARE)],
        k => vec![(0, BASE_SHARE), ((k + 1) / 2, HALF_SHARE), (k, FULL_SHARE)],
    };
    // The first token needs some weight or the phrase could never start, but
    // a context-free nudge must not swamp the acoustics. The further the
    // emitted prefix has matched, the stronger the pull on the next token; a
    // one-token rule is already complete, so it gets its whole weight.
    let progress = |i: usize| {
        if tokens.len() > 1 {
            i as f32 / (tokens.len() - 1) as f32
        } else {
            1.0
        }
    };
    for (matched, share) in tiers {
        let start = context_len - matched;
        for i in 0..tokens.len() {
            let w = weight * share * (PREFIX_FLOOR + (1.0 - PREFIX_FLOOR) * progress(i));
            table
                .entry(full[start..context_len + i].to_vec())
                .or_default()
                .push((tokens[i], w));
        }
    }
    full.len() - 1
}

/// Learned phrases, stored beside the rest of the app's data.
pub fn learn(config: &Config, recognized: &str, corrected: &str) -> Result<Vec<Phrase>> {
    let path = config.bias_path();
    let mut file = BiasFile::load(&path)?;
    let tokenizer = load_tokenizer(config)?;
    let found = correction_phrases(recognized, corrected, &tokenizer);
    if found.is_empty() {
        return Ok(file.phrases);
    }
    for phrase in &found {
        file.add(
            &phrase.to,
            phrase.weight,
            phrase.from.as_deref(),
            phrase.before.as_deref(),
            phrase.after.as_deref(),
        );
    }
    file.save(&path)?;
    log_correction(recognized, corrected, &found).ok();
    Ok(file.phrases)
}

/// The tokenizer the decoder uses, built from the exported vocabulary.
pub fn load_tokenizer(config: &Config) -> Result<Tokenizer> {
    let path = config.models_dir()?.join("vocab.json");
    let vocab: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Tokenizer::new(&vocab))
}

/// Rebuild the phrases from `corrections.jsonl`, which keeps every correction
/// as it was typed. Returns the number of corrections read and the rules.
pub fn rebuild_from_log(path: &Path, tokenizer: &Tokenizer) -> Result<(usize, BiasFile)> {
    #[derive(Deserialize)]
    struct Entry {
        recognized: String,
        corrected: String,
    }
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let mut file = BiasFile::default();
    let mut corrections = 0;
    for line in text.lines() {
        let Ok(entry) = serde_json::from_str::<Entry>(line) else {
            continue;
        };
        corrections += 1;
        for rule in correction_phrases(&entry.recognized, &entry.corrected, tokenizer) {
            file.add(
                &rule.to,
                rule.weight,
                rule.from.as_deref(),
                rule.before.as_deref(),
                rule.after.as_deref(),
            );
        }
    }
    Ok((corrections, file))
}

/// Replacement rules: what the user typed, what was recognized instead, and
/// the text on both sides, all cut on token boundaries.
///
/// The character diff is grouped between the characters it keeps, so an
/// insertion and a deletion that land together become one replacement. Read
/// separately they would store a correction like ディ -> リ as a deleted
/// fragment and a one-character insertion that is too short to keep.
pub fn correction_phrases(recognized: &str, corrected: &str, tokenizer: &Tokenizer) -> Vec<Phrase> {
    let a: Vec<char> = recognized.chars().collect();
    let b: Vec<char> = corrected.chars().collect();

    // Longest common subsequence lengths; the fixups are then read back from
    // the table, preferring an insertion over a deletion on ties.
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    #[derive(Clone, Copy)]
    enum Edit {
        Keep,
        Insert,
        Delete,
    }

    let mut edits = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            edits.push(Edit::Keep);
            i += 1;
            j += 1;
        } else if j < b.len() && (i == a.len() || dp[i][j + 1] >= dp[i + 1][j]) {
            edits.push(Edit::Insert);
            j += 1;
        } else {
            edits.push(Edit::Delete);
            i += 1;
        }
    }

    let spans_a = tokenizer.encode_spans(recognized);
    let spans_b = tokenizer.encode_spans(corrected);

    let mut rules = Vec::new();
    let mut hunk: Option<Hunk> = None;
    let (mut i, mut j) = (0usize, 0usize);
    for edit in &edits {
        match *edit {
            Edit::Keep => {
                if let Some(hunk) = hunk.take() {
                    push_rule(&mut rules, &hunk, &a, &b, &spans_a, &spans_b);
                }
                i += 1;
                j += 1;
            }
            Edit::Insert => {
                let hunk = hunk.get_or_insert(Hunk { to: None, from: None });
                hunk.to = Some(hunk.to.map_or((j, j + 1), |(s, _)| (s, j + 1)));
                j += 1;
            }
            Edit::Delete => {
                let hunk = hunk.get_or_insert(Hunk { to: None, from: None });
                hunk.from = Some(hunk.from.map_or((i, i + 1), |(s, _)| (s, i + 1)));
                i += 1;
            }
        }
    }
    if let Some(hunk) = hunk {
        push_rule(&mut rules, &hunk, &a, &b, &spans_a, &spans_b);
    }
    rules
}

/// One stretch of the diff between two kept characters.
struct Hunk {
    /// Range in the corrected text and in the recognized text.
    to: Option<(usize, usize)>,
    from: Option<(usize, usize)>,
}

/// How many tokens of context are kept on each side.
const CONTEXT_TOKENS: usize = 4;

fn push_rule(
    out: &mut Vec<Phrase>,
    hunk: &Hunk,
    a: &[char],
    b: &[char],
    spans_a: &[(usize, usize, usize)],
    spans_b: &[(usize, usize, usize)],
) {
    let to_range = hunk.to.map_or((0, 0), |range| align(spans_b, range));
    let from_range = hunk.from.map_or((0, 0), |range| align(spans_a, range));

    let to: String = b[to_range.0..to_range.1].iter().collect();
    let from: String = a[from_range.0..from_range.1].iter().collect();
    let (before, after) = if hunk.to.is_some() {
        around(spans_b, b, to_range)
    } else {
        around(spans_a, a, from_range)
    };

    let to = to.trim().to_string();
    let from = from.trim().to_string();
    if to.is_empty() && from.is_empty() {
        return;
    }
    let meaningful = |text: &str| {
        let punctuation = "、。，．！？!?「」『』（）()・…—ー";
        !text.is_empty()
            && !text
                .chars()
                .all(|c| c.is_whitespace() || c.is_ascii_punctuation() || punctuation.contains(c))
    };
    if !meaningful(&to) && !meaningful(&from) {
        return;
    }
    // A lone one-character run is noise; a one-character replacement is not,
    // because the other side pins it down.
    if (to.is_empty() || from.is_empty()) && to.chars().count() + from.chars().count() < 2 {
        return;
    }
    out.push(Phrase {
        to,
        from: if from.is_empty() { None } else { Some(from) },
        weight: 1.0,
        before,
        after,
    });
}

/// Widen a character range to whole tokens, so that the stored text starts and
/// ends where the tokens do.
fn align(spans: &[(usize, usize, usize)], range: (usize, usize)) -> (usize, usize) {
    let (start, end) = range;
    let mut widened = (start, end);
    for &(_, token_start, token_end) in spans {
        if token_start < end && token_end > start {
            widened.0 = widened.0.min(token_start);
            widened.1 = widened.1.max(token_end);
        }
    }
    widened
}

/// The text of up to [`CONTEXT_TOKENS`] whole tokens on each side of a range.
fn around(
    spans: &[(usize, usize, usize)],
    text: &[char],
    range: (usize, usize),
) -> (Option<String>, Option<String>) {
    let (start, end) = range;
    let first = spans.iter().position(|&(_, s, e)| s < end && e > start);
    let last = spans.iter().rposition(|&(_, s, e)| s < end && e > start);
    let (Some(first), Some(last)) = (first, last) else {
        return (None, None);
    };

    let before = if first == 0 {
        None
    } else {
        let from = first.saturating_sub(CONTEXT_TOKENS);
        let at = spans[from].1;
        let text: String = text[at..spans[first].1].iter().collect();
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_string())
    };
    let after = if last + 1 >= spans.len() {
        None
    } else {
        let until = (last + CONTEXT_TOKENS).min(spans.len() - 1);
        let text: String = text[spans[last].2..spans[until].2].iter().collect();
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_string())
    };
    (before, after)
}

/// Append the pair to `corrections.jsonl`, for inspection and for a later
/// fine-tuning pass. Failure to log never blocks the correction itself.
fn log_correction(recognized: &str, corrected: &str, phrases: &[Phrase]) -> Result<()> {
    #[derive(Serialize)]
    struct Entry<'a> {
        at: u64,
        recognized: &'a str,
        corrected: &'a str,
        phrases: &'a [Phrase],
    }
    let dir = Config::data_dir();
    std::fs::create_dir_all(&dir)?;
    let at = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let line = serde_json::to_string(&Entry { at, recognized, corrected, phrases })?;
    let path = dir.join("corrections.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    use std::io::Write;
    writeln!(file, "{line}").with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Tokenizer {
        let vocab: Vec<String> = [
            "<unk>", "\u{2581}", "a", "b", "c", "ab", "abc", "\u{2581}ab", "\u{2581}abc", "x",
            "y", "\u{2581}x",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        Tokenizer::new(&vocab)
    }

    #[test]
    fn bpe_prefers_the_lowest_piece_index() {
        let t = tokenizer();
        // ▁ a b c -> ▁abc is the lowest-index merge chain.
        assert_eq!(t.encode("abc"), vec![8]);
    }

    #[test]
    fn bpe_keeps_unknown_characters_as_unk() {
        let t = tokenizer();
        // The leading ▁ is a real piece; z has no piece at all.
        assert_eq!(t.encode("z"), vec![1, 0]);
    }

    #[test]
    fn a_replacement_is_one_rule_on_both_sides() {
        let t = tokenizer();
        let rules = correction_phrases("エヌビディアのGPU", "NVIDIAのGPU", &t);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].to, "NVIDIA");
        assert_eq!(rules[0].from.as_deref(), Some("エヌビディア"));
        assert!(rules[0].weight > 0.0);
    }

    #[test]
    fn a_one_character_replacement_keeps_both_sides() {
        let t = tokenizer();
        let rules = correction_phrases("タイディング", "タイリング", &t);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].to, "リ");
        assert_eq!(rules[0].from.as_deref(), Some("ディ"));
        assert_eq!(rules[0].before.as_deref(), Some("タイ"));
        assert_eq!(rules[0].after.as_deref(), Some("ング"));
    }

    #[test]
    fn identical_text_learns_nothing() {
        let t = tokenizer();
        assert!(correction_phrases("同じ文", "同じ文", &t).is_empty());
    }

    #[test]
    fn a_case_rule_applies_anywhere_but_not_inside_a_word() {
        let t = tokenizer();
        let phrases = vec![Phrase {
            to: "LB".into(),
            from: Some("lb".into()),
            weight: 1.0,
            before: Some("ーラーの".into()),
            after: None,
        }];
        let compiled = CompiledBias::compile(&phrases, &t, 1.0);
        // The learned context is not there, but a case change means the same.
        assert_eq!(compiled.rewrite("ちゃんとlbとか"), "ちゃんとLBとか");
        // Inside another Latin word it is not the same thing.
        assert_eq!(compiled.rewrite("album"), "album");
    }

    #[test]
    fn a_contextual_rule_needs_its_context_in_the_text() {
        let t = tokenizer();
        let phrases = vec![Phrase {
            to: "リ".into(),
            from: Some("ディ".into()),
            weight: 1.0,
            before: Some("タイ".into()),
            after: None,
        }];
        let compiled = CompiledBias::compile(&phrases, &t, 1.0);
        assert_eq!(compiled.rewrite("タイディング"), "タイリング");
        assert_eq!(compiled.rewrite("デディ"), "デディ");
    }

    #[test]
    fn a_word_corrected_in_several_sentences_applies_anywhere() {
        let t = tokenizer();
        let phrases: Vec<Phrase> = ["古い", "もともと", "ふむ"]
            .iter()
            .map(|before| Phrase {
                to: "cron".into(),
                from: Some("空論".into()),
                weight: 1.0,
                before: Some((*before).into()),
                after: Some("は".into()),
            })
            .collect();
        let compiled = CompiledBias::compile(&phrases, &t, 1.0);
        // None of the recorded surroundings is here, and the word still is.
        assert_eq!(compiled.rewrite("空論が止まってる"), "cronが止まってる");
    }

    #[test]
    fn the_same_sentence_learned_twice_is_still_one_sentence() {
        let t = tokenizer();
        let phrases: Vec<Phrase> = (0..3)
            .map(|_| Phrase {
                to: "cron".into(),
                from: Some("空論".into()),
                weight: 1.0,
                before: Some("古い".into()),
                after: Some("は".into()),
            })
            .collect();
        let compiled = CompiledBias::compile(&phrases, &t, 1.0);
        assert_eq!(compiled.rewrite("空論が止まってる"), "空論が止まってる");
    }

    #[test]
    fn a_word_corrected_once_keeps_to_its_sentence() {
        let t = tokenizer();
        let phrases = vec![Phrase {
            to: "lite".into(),
            from: Some("ライト".into()),
            weight: 1.0,
            before: Some("通にsq".into()),
            after: Some("とか".into()),
        }];
        let compiled = CompiledBias::compile(&phrases, &t, 1.0);
        // 「ライト」 is a word of its own, so one correction stays put.
        assert_eq!(compiled.rewrite("ライトを押す"), "ライトを押す");
        assert_eq!(compiled.rewrite("普通にsqライトとか"), "普通にsqliteとか");
    }

    #[test]
    fn bias_fires_on_a_prefix() {
        let t = tokenizer();
        // "xy" is ▁x + y as a word and x + y inside a sentence.
        assert_eq!(t.encode("xy"), vec![11, 10]);
        assert_eq!(t.encode_context("xy"), vec![9, 10]);
        let phrases = vec![Phrase {
            to: "xy".into(),
            from: None,
            weight: 2.0,
            before: None,
            after: None,
        }];
        let compiled = CompiledBias::compile(&phrases, &t, 1.0);
        let mut bonus = vec![0.0f32; 12];
        compiled.fill(&[], &mut bonus);
        // The phrase boosts the piece the model emits inside a sentence, not
        // the word-start form that only appears at the beginning.
        assert!((bonus[9] - 2.0 * PREFIX_FLOOR).abs() < 1e-6);
        assert_eq!(bonus[11], 0.0);
        compiled.fill(&[9], &mut bonus);
        // Once the prefix matches, the next token gets the full weight.
        assert_eq!(bonus[10], 2.0);
        // The phrase may also start again right after any token.
        assert!((bonus[9] - 2.0 * PREFIX_FLOOR).abs() < 1e-6);
    }

    #[test]
    fn a_different_context_still_pulls_a_little() {
        let t = tokenizer();
        let phrases = vec![Phrase {
            to: "xy".into(),
            from: None,
            weight: 2.0,
            before: Some("c".into()),
            after: None,
        }];
        let compiled = CompiledBias::compile(&phrases, &t, 1.0);
        let mut bonus = vec![0.0f32; 12];

        // No context yet: the phrase pulls, but only by its base share.
        compiled.fill(&[], &mut bonus);
        let none = bonus[9];
        assert!(none > 0.0);
        assert!((none - 2.0 * PREFIX_FLOOR * BASE_SHARE).abs() < 1e-6);

        // A different predecessor pulls exactly as much.
        compiled.fill(&[2], &mut bonus);
        assert!((bonus[9] - none).abs() < 1e-6);

        // "c" is the context, so the first phrase token is at full strength.
        compiled.fill(&[4], &mut bonus);
        assert!(bonus[9] > none);
        assert!((bonus[9] - 2.0 * PREFIX_FLOOR).abs() < 1e-6);

        // Context and first token: the rest of the phrase gets full weight.
        compiled.fill(&[4, 9], &mut bonus);
        assert_eq!(bonus[10], 2.0);
    }

    #[test]
    fn the_learned_continuation_ranks_finished_hypotheses() {
        let t = tokenizer();
        let phrases = vec![Phrase {
            to: "xy".into(),
            from: None,
            weight: 2.0,
            before: Some("c".into()),
            after: Some("a".into()),
        }];
        let compiled = CompiledBias::compile(&phrases, &t, 1.0);
        // c x y a continues the way the correction did; c x y b does not.
        assert!(compiled.right_bonus(&[4, 9, 10, 2]) > 0.0);
        assert_eq!(compiled.right_bonus(&[4, 9, 10, 3]), 0.0);
        // A negative phrase never adds a continuation bonus.
        let negative = vec![Phrase {
            to: "xy".into(),
            from: None,
            weight: -1.0,
            before: Some("c".into()),
            after: Some("a".into()),
        }];
        let compiled = CompiledBias::compile(&negative, &t, 1.0);
        assert_eq!(compiled.right_bonus(&[4, 9, 10, 2]), 0.0);
    }
}
