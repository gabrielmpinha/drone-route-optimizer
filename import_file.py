import pandas as pd
from tkinter import Tk, filedialog, Label, Button, Text, Scrollbar, END

class Pacote:
    def __init__(self, nome, x, y, peso):
        self.nome = nome
        self.x = x
        self.y = y
        self.peso = peso

    def __repr__(self):
        return f"Pacote(Nome={self.nome}, X={self.x}, Y={self.y}, Peso={self.peso})"

def importar_planilha(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo, dtype={"Nome": str, "X": float, "Y": float, "Peso": float})

    colunas_esperadas = ["Nome", "X", "Y", "Peso"]
    if not all(coluna in df.columns for coluna in colunas_esperadas):
        raise ValueError(f"A planilha deve conter as colunas: {colunas_esperadas}")

    lista_pacotes = []
    for _, row in df.iterrows():
        pacote = Pacote(
            nome=row["Nome"],
            x=row["X"],
            y=row["Y"],
            peso=row["Peso"]
        )
        lista_pacotes.append(pacote)

    return lista_pacotes

def selecionar_arquivo():
    caminho_arquivo = filedialog.askopenfilename(
        title="Selecione a planilha",
        filetypes=[("Arquivos Excel", "*.xlsx *.xls")]
    )
    return caminho_arquivo

def exibir_interface():
    def carregar_arquivo():
        caminho_arquivo = selecionar_arquivo()
        if caminho_arquivo:
            try:
                pacotes = importar_planilha(caminho_arquivo)
                texto_resultado.delete(1.0, END)  # Limpa o texto anterior
                texto_resultado.insert(END, "Pacotes importados:\n")
                for pacote in pacotes:
                    texto_resultado.insert(END, f"{pacote}\n")
            except Exception as e:
                texto_resultado.delete(1.0, END)
                texto_resultado.insert(END, f"Erro ao importar a planilha: {e}")
        else:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, "Nenhum arquivo foi selecionado.")

    # Cria a janela principal
    root = Tk()
    root.title("Otimizador de Rotas de Drones")

    # Adiciona um rótulo
    label = Label(root, text="Clique no botão abaixo para selecionar a planilha:")
    label.pack(pady=10)

    # Adiciona um botão para selecionar o arquivo
    botao_selecionar = Button(root, text="Selecionar Planilha", command=carregar_arquivo)
    botao_selecionar.pack(pady=5)

    # Adiciona uma área de texto para exibir os resultados
    texto_resultado = Text(root, height=20, width=60)
    texto_resultado.pack(pady=10)

    # Adiciona uma barra de rolagem à área de texto
    scrollbar = Scrollbar(root, command=texto_resultado.yview)
    texto_resultado.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    # Inicia o loop da interface gráfica
    root.mainloop()

if __name__ == "__main__":
    exibir_interface()