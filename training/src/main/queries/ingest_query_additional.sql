SELECT 
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(program_longsynopsis, " ")) LIMIT {{ TOKEN_LIMIT }}), " ") as synopsis,
  keywords AS tokens, tags
FROM `{{ DATA_SOURCE_TABLE }}`

UNION ALL

SELECT
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(TitleDetails_longsynopsis, " ")) LIMIT {{ TOKEN_LIMIT }}), " ") as synopsis,
  tokens, labels AS tags
FROM `res-nbcupea-dev-ds-sandbox-001.metadata_enhancement.test_titles_synopsis_keywords_all_labels`

{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}