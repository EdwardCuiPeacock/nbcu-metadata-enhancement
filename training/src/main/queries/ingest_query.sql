SELECT 
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(program_longsynopsis, " ")) LIMIT {{ TOKEN_LIMIT }}), " ") as synopsis,
  tags
FROM `{{ DATA_SOURCE_TABLE }}`
{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}