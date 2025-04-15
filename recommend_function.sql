DROP FUNCTION recommend_recipe(sampleRecipeId int, numResults int);
CREATE OR REPLACE FUNCTION
    recommend_recipe(sampleRecipeId int, numResults int)
returns table(
            in_recipeName character varying,
			out_recipeName character varying,
            out_nutrition character varying,
            out_similarityScore real)
as $$
declare
    queryEmbedding vector(1536);
	sampleRecipename character varying ;
    sampleRecipeText character varying ;
begin
	sampleRecipename := (select
                            recipe_name
                        from
                            recipes where rid = sampleRecipeId);
						
    sampleRecipeText := (select
                            recipe_name||' '||cuisine_path||' '||ingredients||' '||nutrition||' '||directions
                        from
                            recipes where rid = sampleRecipeId);

    queryEmbedding := (azure_openai.create_embeddings('text-embedding-ada-002',sampleRecipeText));

    return query
    select
        distinct sampleRecipename,r.recipe_name,
        r.nutrition,
        (r.recipe_vector <=> queryEmbedding)::real as score
    from
        recipes r
    order by score asc limit numResults; -- cosine distance
end $$
language plpgsql;

/*
--test
select in_recipename, out_recipename, out_similarityscore from recommend_recipe(1, 20)
WHERE out_similarityscore !=0
ORDER BY 2 DESC
; 
-- search for 20 recipe recommendations that closest to recipeId 1
*/


DROP FUNCTION recommend_recipe_by_description(recipeDescription character varying, numResults int);
CREATE OR REPLACE FUNCTION
    recommend_recipe_by_description(recipeDescription character varying, numResults int)
returns table(
            in_recipeDescription character varying,
			out_recipeName character varying,
            out_nutrition character varying,
            out_similarityScore real)
as $$
declare
    queryEmbedding vector(1536);
	
begin
	
    queryEmbedding := (azure_openai.create_embeddings('text-embedding-ada-002',recipeDescription));

    return query
    select
        distinct recipeDescription,r.recipe_name,
        r.nutrition,
        (r.recipe_vector <=> queryEmbedding)::real as score
    from
        recipes r
    order by score asc limit numResults; -- cosine distance
end $$
language plpgsql;

/*
--Test
select in_recipedescription, out_recipename, out_similarityscore from recommend_recipe_by_description('chicken and peanut', 20)
WHERE out_similarityscore !=0
ORDER BY 2 DESC
; 

*/