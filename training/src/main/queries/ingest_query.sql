SELECT 
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(program_longsynopsis, " ")) LIMIT {{ TOKEN_LIMIT }}), " ") as synopsis,
  tokens, tags
FROM `{{ GOOGLE_CLOUD_PROJECT }}.metadata_enhancement.meta_synopsis_100tag_with_token_edc_dev`
{% if TEST_LIMIT -%}
   LIMIT {{ TEST_LIMIT }}
{% endif %}