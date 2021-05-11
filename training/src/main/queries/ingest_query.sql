SELECT 
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(program_longsynopsis, " ")) LIMIT {{ TOKEN_LIMIT }}), " ") as synopsis,
  STRING_AGG(kk, " ") AS keywords, ANY_VALUE(tags) AS tags
FROM `{{ DATA_SOURCE_TABLE }}`,
UNNEST(keywords) kk
GROUP BY program_title, program_type, program_language, program_longsynopsis
{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}