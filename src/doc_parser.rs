//! Documentation parser for NeuroScript
//!
//! Parses triple-slash doc comments into structured Documentation objects.
//! Supports markdown-style sections like `# Parameters`, `# Shape Contract`, etc.

use crate::interfaces::Documentation;
use miette::SourceSpan;
use std::collections::HashMap;

/// Parse a vec of doc comment lines into structured Documentation
///
/// Doc comments follow this format:
/// ```
/// /// Brief one-line description
/// ///
/// /// Longer description paragraph.
/// ///
/// /// # Parameters
/// /// - dim: Model dimension
/// /// - expansion: Expansion factor
/// ///
/// /// # Shape Contract
/// /// - Input: [*batch, seq, dim]
/// /// - Output: [*batch, seq, dim]
/// ```
pub fn parse_doc_comments(lines: Vec<String>, span: Option<SourceSpan>) -> Documentation {
    if lines.is_empty() {
        return Documentation::empty();
    }

    // Strip the "///" prefix from each line and trim whitespace
    let stripped_lines: Vec<String> = lines
        .iter()
        .map(|line| {
            line.strip_prefix("///")
                .unwrap_or(line)
                .trim_start()
                .to_string()
        })
        .collect();

    // Join all lines into raw content
    let content = stripped_lines.join("\n").trim().to_string();

    // Parse sections (lines starting with "# SectionName")
    let sections = parse_sections(&stripped_lines);

    Documentation {
        content,
        sections,
        span,
    }
}

/// Parse markdown-style sections from doc comment lines
///
/// Sections are marked by lines starting with "# " (markdown heading)
/// For example:
/// - "# Parameters" starts the parameters section
/// - "# Shape Contract" starts the shape contract section
/// - "# Example" starts the example section
fn parse_sections(lines: &[String]) -> HashMap<String, String> {
    let mut sections = HashMap::new();
    let mut current_section: Option<String> = None;
    let mut current_content: Vec<String> = Vec::new();

    for line in lines {
        // Check if this line starts a new section
        if let Some(section_name) = line.strip_prefix("# ") {
            // Save previous section if exists
            if let Some(section) = current_section.take() {
                let content = current_content.join("\n").trim().to_string();
                if !content.is_empty() {
                    sections.insert(section, content);
                }
                current_content.clear();
            }

            // Start new section
            current_section = Some(section_name.trim().to_string());
        } else if current_section.is_some() {
            // Add line to current section
            current_content.push(line.clone());
        }
        // Lines before any section are ignored (they're part of the main content)
    }

    // Save the last section
    if let Some(section) = current_section {
        let content = current_content.join("\n").trim().to_string();
        if !content.is_empty() {
            sections.insert(section, content);
        }
    }

    sections
}

/// Extract the brief description (first non-empty line before any section)
pub fn extract_brief(doc: &Documentation) -> String {
    for line in doc.content.lines() {
        let trimmed = line.trim();
        // Stop at first section marker
        if trimmed.starts_with("# ") {
            break;
        }
        // Return first non-empty line
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    String::new()
}

/// Extract the detailed description (paragraphs before any section, excluding brief)
pub fn extract_description(doc: &Documentation) -> String {
    let mut lines = Vec::new();
    let mut found_brief = false;

    for line in doc.content.lines() {
        let trimmed = line.trim();

        // Stop at first section marker
        if trimmed.starts_with("# ") {
            break;
        }

        // Skip brief (first non-empty line)
        if !found_brief && !trimmed.is_empty() {
            found_brief = true;
            continue;
        }

        if found_brief {
            lines.push(trimmed.to_string());
        }
    }

    lines.join("\n").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_doc() {
        let lines = vec![
            "/// Brief description".to_string(),
            "///".to_string(),
            "/// Longer description here.".to_string(),
        ];

        let doc = parse_doc_comments(lines, None);
        assert!(!doc.content.is_empty());
        assert_eq!(extract_brief(&doc), "Brief description");
    }

    #[test]
    fn test_parse_sections() {
        let lines = vec![
            "/// Brief description".to_string(),
            "///".to_string(),
            "/// # Parameters".to_string(),
            "/// - dim: Dimension".to_string(),
            "/// - expansion: Factor".to_string(),
            "///".to_string(),
            "/// # Shape Contract".to_string(),
            "/// - Input: [*, dim]".to_string(),
        ];

        let doc = parse_doc_comments(lines, None);
        assert!(doc.sections.contains_key("Parameters"));
        assert!(doc.sections.contains_key("Shape Contract"));

        let params = doc.sections.get("Parameters").unwrap();
        assert!(params.contains("dim: Dimension"));
        assert!(params.contains("expansion: Factor"));
    }

    #[test]
    fn test_extract_brief() {
        let lines = vec![
            "/// Brief one-liner".to_string(),
            "///".to_string(),
            "/// Longer description.".to_string(),
        ];

        let doc = parse_doc_comments(lines, None);
        assert_eq!(extract_brief(&doc), "Brief one-liner");
    }

    #[test]
    fn test_extract_description() {
        let lines = vec![
            "/// Brief one-liner".to_string(),
            "///".to_string(),
            "/// This is a longer description".to_string(),
            "/// that spans multiple lines.".to_string(),
            "///".to_string(),
            "/// # Parameters".to_string(),
            "/// - param: description".to_string(),
        ];

        let doc = parse_doc_comments(lines, None);
        let desc = extract_description(&doc);
        assert!(desc.contains("longer description"));
        assert!(desc.contains("multiple lines"));
        assert!(!desc.contains("Parameters")); // Should stop before sections
    }

    #[test]
    fn test_empty_doc() {
        let doc = parse_doc_comments(vec![], None);
        assert!(doc.content.is_empty());
        assert!(doc.sections.is_empty());
    }
}
