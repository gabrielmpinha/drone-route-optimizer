import pandas as pd
from tkinter import Tk, filedialog, Label, Button, Text, Scrollbar, END, Frame, Entry
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    caminho_arquivo = None

    def carregar_arquivo():
        nonlocal caminho_arquivo
        caminho_arquivo = selecionar_arquivo()
        if caminho_arquivo:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, f"Arquivo selecionado: {caminho_arquivo}\n")
        else:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, "Nenhum arquivo foi selecionado.\n")

    def enviar_dados():
        if not caminho_arquivo:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, "Erro: Nenhum arquivo foi selecionado.\n")
            return

        try:
            x_inicial = float(entry_x.get())
            y_inicial = float(entry_y.get())
        except ValueError:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, "Erro: Valores de X e Y devem ser números.\n")
            return

        try:
            pacotes = importar_planilha(caminho_arquivo)
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, f"Posição inicial: X={x_inicial}, Y={y_inicial}\n")
            texto_resultado.insert(END, "Pacotes importados:\n")
            for pacote in pacotes:
                texto_resultado.insert(END, f"{pacote}\n")
            exibir_grafico(pacotes)
        except Exception as e:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, f"Erro ao importar a planilha: {e}\n")

    def exibir_grafico(pacotes):
        nomes = [pacote.nome for pacote in pacotes]
        pesos = [pacote.peso for pacote in pacotes]

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(nomes, pesos, color='blue')
        ax.set_title("Peso por Pacote")
        ax.set_xlabel("Nome")
        ax.set_ylabel("Peso")

        for widget in frame_grafico.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack()

    root = Tk()
    root.title("Otimizador de Rotas de Drones")

    label = Label(root, text="Clique no botão abaixo para selecionar a planilha:")
    label.pack(pady=10)

    botao_selecionar = Button(root, text="Selecionar Planilha", command=carregar_arquivo)
    botao_selecionar.pack(pady=5)

    label_x = Label(root, text="Posição X inicial:")
    label_x.pack(pady=5)
    entry_x = Entry(root)
    entry_x.insert(0, "0")  # Valor padrão
    entry_x.pack(pady=5)

    label_y = Label(root, text="Posição Y inicial:")
    label_y.pack(pady=5)
    entry_y = Entry(root)
    entry_y.insert(0, "0")  # Valor padrão
    entry_y.pack(pady=5)

    botao_enviar = Button(root, text="Enviar", command=enviar_dados)
    botao_enviar.pack(pady=10)

    texto_resultado = Text(root, height=10, width=60)
    texto_resultado.pack(pady=10)

    scrollbar = Scrollbar(root, command=texto_resultado.yview)
    texto_resultado.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    frame_grafico = Frame(root)
    frame_grafico.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    exibir_interface()