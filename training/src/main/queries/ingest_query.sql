SELECT 
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(program_longsynopsis, " ")) LIMIT {{ token_limit }}), " ") as synopsis,
  tags
FROM `{{ project }}.{{ dataset }}.{{ table }}`
{% if limit -%}
   LIMIT {{ limit }}
{% endif %}