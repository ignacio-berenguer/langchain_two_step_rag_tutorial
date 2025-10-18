from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from db import get_vector_store, get_embeddings

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Herramienta para recuperar información relevante para ayudarte a responder cualquier consulta utilizando el contenido de los documentos."""
    
    # Here you would implement the logic to query your vector store
    # and retrieve the relevant documents based on the input query.
    # This is a placeholder implementation.
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_model(query):
    tools = [retrieve_context]
    # If desired, specify custom instructions
    prompt = (
        "Tienes acceso a una herramienta que te permite recuperar información de libros y documentos. "
        "Usa la herramienta para responder a las preguntas de los usuarios."
    )

    model = init_chat_model("openai:gpt-4.1")
    agent = create_agent(model, tools, system_prompt=prompt)
    # for event in agent.stream(
    #     {"messages": [{"role": "user", "content":query}]},
    #     stream_mode="values"
    # ):
    #     event["messages"][-1].pretty_print()
    result = agent.invoke( {"messages": [{"role": "user", "content":query}]})
    return result

def main():
    print("Consulta a la base de datos de documentos")
    
    # "Háblame de las emboscadas que le hacen al capitán Alatriste. Quiero una respuesta muy detallada, si le hacen varias emboscadas, quiero conocerlas todas\n\n"
    # "Una vez que consigas la respuesta, quiero que me describas a los personajes que participan en las emboscadas\n\n"
    # "Luego, sin utilizar la herramienta de consulta quiero que hagas un resumen histórico de la época en la que suceden los hechos de la novela."
    #"En el libro hay varias poesías, quiero conocerlas todas y dime en qué página del libro aparecen y el personaje que las dice"
    # "Qué piensa Alatriste de que Iñigo Balboa esté enamorado?"
    # "Cómo es la personalidad de la chica de la que se enamora Iñigo Balboa. Quiero una respuesta muy detallada, con ejemplos del texto y las páginas en las que se menciona"
    # "Quiero que busques los 10 personajes más importantes de la novela, y que para cada uno de ellos utilices la herramienta para describir sus rasgos de carácter más importantes, y su participación en la novela. Quiero una respuesta detallada, en la que menciones la página en la que te basas para las afirmaciones"
    # "Dame todas las páginas en las que aparece Iñigpo de Balboa y un resumen de lo que sucede. No me basta con una página, así que utiliza la herramienta de consulta para ver todas las páginas en las que aparece"
    # "Quienes son los amigos de Alatriste"
    # "En qué artículo del Convenio XXI de la INdustria Química se menciona la obligación de informar las altas directas al Comité de Empresa, excluyendo a las empresas de trabajo temporal. "

    while True:
        query = input("\n🔍 Developer Tools Query: ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        result = query_model(query)
        if result:
            result["messages"][-1].pretty_print()
            # for msg in result["messages"]:
            #     msg.pretty_print()

    
if __name__ == "__main__":
    main()