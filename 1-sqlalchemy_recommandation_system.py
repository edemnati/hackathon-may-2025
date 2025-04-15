import json
import os
from pathlib import Path
import pandas as pd

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# Define the models
class Base(DeclarativeBase):
    pass


class Recipes(Base):
    __tablename__ = "recipes"
    rid: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    recipe_name: Mapped[str] = mapped_column()
    prep_time: Mapped[str] = mapped_column()
    cook_time: Mapped[str] = mapped_column()
    total_time: Mapped[str] = mapped_column()
    servings: Mapped[str] = mapped_column()
    r_yield: Mapped[str] = mapped_column()
    ingredients: Mapped[str] = mapped_column()
    directions: Mapped[str] = mapped_column()
    rating: Mapped[str] = mapped_column()
    url: Mapped[str] = mapped_column()
    cuisine_path: Mapped[str] = mapped_column()
    nutrition: Mapped[str] = mapped_column()
    timing: Mapped[str] = mapped_column()
    img_src: Mapped[str] = mapped_column()
    recipe_vector = mapped_column(Vector(1536))  # ada-002 is 1536-dimensional



def setup(RUN_SETUP: bool = False):
    """Set up the database and create the tables."""
    
    # Define HNSW index to support vector similarity search through the vector_cosine_ops access method (cosine distance). The SQL operator for cosine distance is written as <=>.
    
    index = Index(
        "hnsw_index_for_cosine_distance_similarity_search",
        Recipes.recipe_vector,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"recipe_vector": "vector_cosine_ops"},
    )

    # Connect to the database based on environment variables
    load_dotenv(".env", override=True)
    POSTGRES_HOST = os.environ["POSTGRES_HOST"]
    POSTGRES_USERNAME = os.environ["POSTGRES_USERNAME"]
    POSTGRES_DATABASE = os.environ["POSTGRES_DATABASE"]

    if POSTGRES_HOST.endswith(".database.azure.com"):
        print("Authenticating to Azure Database for PostgreSQL using Azure Identity...")
        azure_credential = DefaultAzureCredential()
        token = azure_credential.get_token("https://ossrdbms-aad.database.windows.net/.default")
        POSTGRES_PASSWORD = token.token
    else:
        POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]

    DATABASE_URI = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DATABASE}"
    # Specify SSL mode if needed
    if POSTGRES_SSL := os.environ.get("POSTGRES_SSL"):
        DATABASE_URI += f"?sslmode={POSTGRES_SSL}"

    engine = create_engine(DATABASE_URI, echo=False)

    # Create pgvector extension
    if RUN_SETUP:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS azure_ai"))
            conn.execute(text(f"select azure_ai.set_setting('azure_openai.endpoint','{os.environ.get("AZURE_OPENAI_ENDPOINT")}');"))
            conn.execute(text(f"select azure_ai.set_setting('azure_openai.subscription_key', '{os.environ.get("AZURE_OPENAI_KEY")}');"))
        
        # Drop all tables (and indexes) defined in this model from the database, if they already exist
        Base.metadata.drop_all(engine)
        # Create all tables (and indexes) defined for this model in the database
        Base.metadata.create_all(engine)

    return engine


def main():
    RUN_SETUP = True
    print("Starting the SQLAlchemy recommendation system...")
    engine = setup(RUN_SETUP)
    print("Database setup complete.")
    # Insert data and issue queries
    with Session(engine) as session:
        if RUN_SETUP:
            #Delete all data in the table
            session.execute(text("DELETE FROM recipes;"))
            session.commit()

            print("Session started.")
            # Insert the Recipes from the JSON file
            data_path = "./recipes.csv"
            df_receipes = pd.read_csv(data_path)
            
            df_receipes = df_receipes.astype(str).fillna("")
            print(f"Data loaded from CSV. length: ", len(df_receipes))
            ct = 0
            for _,row in df_receipes.iterrows():
                #embedding = azure_openai.create_embeddings('text-embedding-ada-002', f"{row['recipe_name']} {row['cuisine_path']} {row['ingredients']} {row['nutrition']} {row['directions']}")
                receipe = Recipes(
                                #rid=row['Unnamed: 0'],
                                recipe_name=row['recipe_name'],
                                prep_time=row['prep_time'],
                                cook_time=row['cook_time'],
                                total_time=row['total_time'],
                                servings=row['servings'],
                                r_yield=row['yield'],
                                ingredients=row['ingredients'],
                                directions=row['directions'],
                                rating=row['rating'],
                                url=row['url'],
                                cuisine_path=row['cuisine_path'],
                                nutrition=row['nutrition'],
                                timing=row['timing'],
                                img_src=row['img_src'],
                                #recipe_vector=embedding
                                )
                session.add(receipe)
                
                ct+=1
                if ct % 100 == 0:
                    print(f"Try to commit at {ct} recipes.")
                    session.commit()
                    print(f"Inserted {ct} recipes.")
            session.commit()

            RUN_EMBEDDINGS = True
            if RUN_EMBEDDINGS:
                #Add embeddings 
                with engine.begin() as conn:
                    sql_satement = """WITH ro AS (
                                    SELECT ro.rid
                                    FROM
                                        recipes ro
                                    WHERE
                                        ro.recipe_vector is null
                                        LIMIT 5000
                                )
                                UPDATE
                                    recipes r
                                SET
                                    recipe_vector = azure_openai.create_embeddings('text-embedding-ada-002', r.recipe_name||' '||r.cuisine_path||' '||r.ingredients||' '||r.nutrition||' '||r.directions)
                                FROM
                                    ro
                                WHERE
                                    r.rid = ro.rid;"""
                    conn.execute(text(sql_satement))

    # Query for target Recipe, the one whose title matches "Winnie the Pooh"
    query = select(Recipes).where(Recipes.recipe_name == "Apple Pie by Grandma Ople")
    target_recipe = session.execute(query).scalars().first()
    if target_recipe is None:
        print("Receipe not found")
        exit(1)

    # Find the 5 most similar Recipes to "Winnie the Pooh"
    most_similars = session.execute(text("""SELECT  * 
                                         FROM recipes 
                                         WHERE recipe_vector IS NOT NULL 
                                            AND rid != :rid ORDER BY recipe_vector <=> :recipe_vector LIMIT 5"""
                                         ), 
                                    {"rid": target_recipe.rid, "recipe_vector": json.dumps(target_recipe.recipe_vector.tolist())}).all()
    
    '''
    most_similars = session.scalars(select(Recipes).order_by(Recipes.recipe_vector.cosine_distance(target_recipe.recipe_vector)).limit(10))
    '''
    print(f"Five most similar recipes to '{target_recipe.recipe_name}':")
    for Recipe in most_similars:
        print(f"\t{Recipe.recipe_name}\t{Recipe.prep_time}")

#Run script
if __name__ == "__main__":
    main()
    print("Script executed successfully.")