-- TODO: Template and refactor 
WITH titles AS (
     SELECT 
         DISTINCT  
         COALESCE(InSeasonSeries_Id, TitleId) as InSeasonSeries_Id,
         TitleDetails_LongSynopsis,
         InSeasonSeries_Tags,
         TitleTags,
         TitleSubgenres,
         TitleType
     FROM 
         `ml-sandbox-101.metadata_sky.ContentMetadataView`
 ),
 melted AS (
     SELECT 
         DISTINCT 
         InSeasonSeries_Id,
         TitleDetails_LongSynopsis,
         TitleType,
         TRIM(tags) as tags 
     FROM (
         SELECT 
             DISTINCT
             InSeasonSeries_Id,
             TitleDetails_LongSynopsis,
             TitleType,
             tags 
         FROM titles
         CROSS JOIN UNNEST(SPLIt(InSeasonSeries_Tags,',')) tags
         UNION ALL
         SELECT 
             DISTINCT 
             InSeasonSeries_Id,
             TitleDetails_LongSynopsis,
             TitleType,
             tags
         FROM 
             titles
         CROSS JOIN UNNEST(SPLIt(TitleSubgenres,',')) tags
         UNION ALL
         SELECT 
             DISTINCT  
             InSeasonSeries_Id,
             TitleDetails_LongSynopsis,
             TitleType,
             tags
         FROM 
             titles
         CROSS JOIN UNNEST(SPLIt(TitleTags,',')) tags
     )
     WHERE tags <> ''
 ),
 subgenres AS (
     SELECT TRIM(tags) as tags FROM(
         SELECT 
         DISTINCT tags
         FROM `ml-sandbox-101.metadata_sky.ContentMetadataView`
         CROSS JOIN UNNEST(SPLIt(TitleSubgenres,',')) tags
     )
     WHERE tags <> ''
 ),
 with_labels AS (
     SELECT 
         InSeasonSeries_Id,
         TitleType,
         ARRAY_TO_STRING(ARRAY_AGG(TitleDetails_LongSynopsis), " ") as synopsis,
         ARRAY_AGG(melted.tags) as labels
     FROM melted 
     LEFT JOIN subgenres
     ON melted.tags = subgenres.tags
     WHERE subgenres.tags IS NOT NULL
     OR melted.tags IN (
         'not for kids',
         'older teens (ages 15+)',
         'teens (ages 13-14)',
         'tweens (ages 10-12)',
         'big kids (ages 8-9)',
         'little kids (ages 5-7)',
         'preschoolers (ages 2-4)'
     )
     GROUP BY 
         InSeasonSeries_Id,
         TitleType
 ),
 with_tags AS (
     SELECT
         InSeasonSeries_Id,
         ARRAY_AGG(tags) as tags 
     FROM melted
     GROUP BY
     InSeasonSeries_Id
 )
 
 SELECT
     with_labels.InSeasonSeries_Id as content_id,
     TitleType,
     synopsis,
     ARRAY(SELECT DISTINCT label FROM UNNEST(labels) AS label) as labels,
     ARRAY(SELECT DISTINCT tag FROM UNNEST(tags) as tag) as tags
 FROM with_labels
 INNER JOIN with_tags
     ON with_labels.InSeasonSeries_Id = with_tags.InSeasonSeries_Id
 LIMIT 100