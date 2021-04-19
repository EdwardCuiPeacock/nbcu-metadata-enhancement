CREATE TEMP FUNCTION dedup(val ANY TYPE) AS ((
  SELECT ARRAY_AGG(t)
  FROM (SELECT DISTINCT * FROM UNNEST(val) v) t
)); -- deduplicate tags

SELECT 
  ARRAY_TO_STRING(ARRAY(
    SELECT * 
        FROM UNNEST(SPLIT(program_longsynopsis, " ")) LIMIT {{ token_limit }}), " ") as synopsis,
  dedup(tags) AS tags
FROM `{{ project }}.{{ dataset }}.{{ table }}`
{% if limit -%}
   LIMIT {{ limit }}
{% endif %}