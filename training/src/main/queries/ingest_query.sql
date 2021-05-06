SELECT 
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(program_longsynopsis, " ")) LIMIT {{ TOKEN_LIMIT }}), " ") as synopsis,
  LOWER(program_title) AS title, tags
FROM `{{ DATA_SOURCE_TABLE }}`
{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}