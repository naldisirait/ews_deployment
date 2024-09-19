from fastapi import FastAPI

app = FastAPI()

# Define a simple route to get a greeting
@app.get("/greet")
def greet(name: str = "World"):
    return {"message": f"Hello, {name}!"}

# This is me
