#!/usr/bin/env python3
"""
Neo4j Queries Collection

Chứa tất cả các Cypher queries được sử dụng trong GraphRetriever
"""

# Document validation queries
VALIDATE_DOCUMENT_QUERY = """
MATCH (d:Document {file_name: $document_name})
RETURN count(d) as document_count
"""

# Language detection queries
LANGUAGE_DETECTION_SCOPED_QUERY = """
MATCH (doc:Document {file_name: $document_name})-[:CONTAINS*1..3]->(n)
WHERE coalesce(n.text, n.name, n.summary, n.description) IS NOT NULL
WITH n, coalesce(n.text, n.name, n.summary, n.description) as content
WHERE size(content) > 10
RETURN content
ORDER BY rand()
LIMIT $sample_size
"""

LANGUAGE_DETECTION_GLOBAL_QUERY = """
MATCH (n)
WHERE coalesce(n.text, n.name, n.summary, n.description) IS NOT NULL
WITH n, coalesce(n.text, n.name, n.summary, n.description) as content
WHERE size(content) > 10
RETURN content
ORDER BY rand()
LIMIT $sample_size
"""

# Hybrid search queries
HYBRID_SEARCH_SCOPED_QUERY = """
WITH $embedding AS inputEmbedding, $keywords AS searchKeywords
MATCH (doc:Document {file_name: $document_name})-[:CONTAINS*1..3]->(n)
WHERE n.embedding IS NOT NULL

// Semantic similarity score
CALL {
  WITH n, inputEmbedding
  RETURN gds.similarity.cosine(n.embedding, inputEmbedding) AS semantic_score
}

// Keyword matching score
WITH n, semantic_score,
     CASE 
       WHEN any(keyword IN searchKeywords WHERE 
           toLower(coalesce(n.text, '')) CONTAINS toLower(keyword) OR
           toLower(coalesce(n.name, '')) CONTAINS toLower(keyword) OR
           toLower(coalesce(n.summary, '')) CONTAINS toLower(keyword) OR
           toLower(coalesce(n.description, '')) CONTAINS toLower(keyword))
       THEN 0.3
       ELSE 0.0
     END AS keyword_score

// Node type bonus (prioritize certain types)
WITH n, semantic_score, keyword_score,
     CASE 
       WHEN 'Document' IN labels(n) THEN 0.1
       WHEN 'Chunk' IN labels(n) THEN 0.05
       ELSE 0.0
     END AS type_bonus

// Content length bonus (prefer substantial content)
WITH n, semantic_score, keyword_score, type_bonus,
     CASE 
       WHEN size(coalesce(n.text, '')) < 50 THEN -0.1
       WHEN size(coalesce(n.text, '')) > 500 THEN 0.05
       ELSE 0.0
     END AS content_bonus

// Combined score
WITH n, semantic_score, keyword_score, type_bonus, content_bonus,
     (semantic_score * 0.6 + keyword_score * 0.3 + type_bonus + content_bonus) AS combined_score

WHERE combined_score >= $min_score

RETURN n AS node, 
       labels(n) AS node_labels,
       semantic_score,
       keyword_score,
       combined_score,
       type_bonus,
       content_bonus
ORDER BY combined_score DESC
LIMIT $limit
"""

HYBRID_SEARCH_GLOBAL_QUERY = """
WITH $embedding AS inputEmbedding, $keywords AS searchKeywords
MATCH (n)
WHERE n.embedding IS NOT NULL

// Semantic similarity score
CALL {
  WITH n, inputEmbedding
  RETURN gds.similarity.cosine(n.embedding, inputEmbedding) AS semantic_score
}

// Keyword matching score
WITH n, semantic_score,
     CASE 
       WHEN any(keyword IN searchKeywords WHERE 
           toLower(coalesce(n.text, '')) CONTAINS toLower(keyword) OR
           toLower(coalesce(n.name, '')) CONTAINS toLower(keyword) OR
           toLower(coalesce(n.summary, '')) CONTAINS toLower(keyword) OR
           toLower(coalesce(n.description, '')) CONTAINS toLower(keyword))
       THEN 0.3
       ELSE 0.0
     END AS keyword_score

// Node type bonus (prioritize certain types)
WITH n, semantic_score, keyword_score,
     CASE 
       WHEN 'Document' IN labels(n) THEN 0.1
       WHEN 'Chunk' IN labels(n) THEN 0.05
       ELSE 0.0
     END AS type_bonus

// Content length bonus (prefer substantial content)
WITH n, semantic_score, keyword_score, type_bonus,
     CASE 
       WHEN size(coalesce(n.text, '')) < 50 THEN -0.1
       WHEN size(coalesce(n.text, '')) > 500 THEN 0.05
       ELSE 0.0
     END AS content_bonus

// Combined score
WITH n, semantic_score, keyword_score, type_bonus, content_bonus,
     (semantic_score * 0.6 + keyword_score * 0.3 + type_bonus + content_bonus) AS combined_score

WHERE combined_score >= $min_score

RETURN n AS node, 
       labels(n) AS node_labels,
       semantic_score,
       keyword_score,
       combined_score,
       type_bonus,
       content_bonus
ORDER BY combined_score DESC
LIMIT $limit
"""

# Parent and related content queries
PARENT_CONTENT_QUERY = """
MATCH (child)-[:HAS_PARENT]->(parent)
WHERE id(child) = $node_id
RETURN parent.text as parent_text, parent.summary as parent_summary
"""

RELATED_CONTENT_QUERY = """
MATCH (node)-[r]-(related)
WHERE id(node) = $node_id AND type(r) IN ['NEXT', 'PREVIOUS', 'SIMILAR_TO', 'RELATED_TO']
RETURN related.text as related_text, related.summary as related_summary, type(r) as relationship_type
LIMIT 3
"""

# Basic semantic search queries
SEMANTIC_SEARCH_SCOPED_QUERY = """
WITH $embedding AS inputEmbedding
MATCH (doc:Document {file_name: $document_name})-[:CONTAINS*1..3]->(n)
WHERE n.embedding IS NOT NULL

CALL {
  WITH n, inputEmbedding
  RETURN gds.similarity.cosine(n.embedding, inputEmbedding) AS similarity_score
}

WHERE similarity_score >= 0.1
RETURN n AS node, labels(n) AS node_labels, similarity_score
ORDER BY similarity_score DESC
LIMIT $limit
"""

SEMANTIC_SEARCH_GLOBAL_QUERY = """
WITH $embedding AS inputEmbedding
MATCH (n)
WHERE n.embedding IS NOT NULL

CALL {
  WITH n, inputEmbedding
  RETURN gds.similarity.cosine(n.embedding, inputEmbedding) AS similarity_score
}

WHERE similarity_score >= 0.1
RETURN n AS node, labels(n) AS node_labels, similarity_score
ORDER BY similarity_score DESC
LIMIT $limit
"""
