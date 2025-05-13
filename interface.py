import pandas as pd
from tkinter import Tk, filedialog, Label, Button, Text, Scrollbar, END, Frame, Entry
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from drone_problem import drone_optimization, Pacote

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
            z=0,  # Z sempre será 0
            peso=[row["Peso"]]
        )
        lista_pacotes.append(pacote)
    return lista_pacotes

# Função para abrir um diálogo de seleção de arquivo
def selecionar_arquivo():
    caminho_arquivo = filedialog.askopenfilename(
        title="Selecione a planilha",
        filetypes=[("Arquivos Excel", "*.xlsx *.xls")]
    )
    return caminho_arquivo

# Função principal para exibir a interface gráfica
def exibir_interface():
    caminho_arquivo = None  # Variável para armazenar o caminho do arquivo selecionado

    # Função para carregar o arquivo selecionado
    def carregar_arquivo():
        nonlocal caminho_arquivo
        caminho_arquivo = selecionar_arquivo()
        if caminho_arquivo:
            texto_resultado.delete(1.0, END)  # Limpa o texto anterior
            texto_resultado.insert(END, f"Arquivo selecionado: {caminho_arquivo}\n")
        else:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, "Nenhum arquivo foi selecionado.\n")

    # Função para processar os dados após o clique no botão "Enviar"
    def enviar_dados():
        if not caminho_arquivo:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, "Erro: Nenhum arquivo foi selecionado.\n")
            return

        try:
            # Obtém os valores de X e Y iniciais das caixas de texto
            x_inicial = float(entry_x.get())
            y_inicial = float(entry_y.get())
            initial_pos = (x_inicial, y_inicial, 0)  # Posição inicial do drone
        except ValueError:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, "Erro: Valores de X e Y devem ser números.\n")
            return

        try:
            # Importa os pacotes da planilha
            pacotes = importar_planilha(caminho_arquivo)

            # Chama a função drone_optimization
            plot_result, solutions = drone_optimization(pacotes, initial_pos)

            # Exibe as soluções na área de texto
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, "Soluções encontradas:\n")
            
            texto_resultado.insert(END, f"{solutions}\n")

            # Exibe o gráfico retornado pela função
            exibir_grafico(plot_result)
        except Exception as e:
            texto_resultado.delete(1.0, END)
            texto_resultado.insert(END, f"Erro ao processar os dados: {e}\n")

    # Função para exibir o gráfico retornado pela função drone_optimization
    def exibir_grafico(plot_result):
        # Remove gráficos anteriores e exibe o novo gráfico
        for widget in frame_grafico.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(plot_result, master=frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack()

    # Cria a janela principal
    root = Tk()
    root.title("Otimizador de Rotas de Drones")

    # Rótulo para instrução
    label = Label(root, text="Clique no botão abaixo para selecionar a planilha:")
    label.pack(pady=10)

    # Botão para selecionar a planilha
    botao_selecionar = Button(root, text="Selecionar Planilha", command=carregar_arquivo)
    botao_selecionar.pack(pady=5)

    # Caixa de texto para a posição X inicial
    label_x = Label(root, text="Posição X inicial:")
    label_x.pack(pady=5)
    entry_x = Entry(root)
    entry_x.insert(0, "0")  # Valor padrão
    entry_x.pack(pady=5)

    # Caixa de texto para a posição Y inicial
    label_y = Label(root, text="Posição Y inicial:")
    label_y.pack(pady=5)
    entry_y = Entry(root)
    entry_y.insert(0, "0")  # Valor padrão
    entry_y.pack(pady=5)

    # Botão para enviar os dados
    botao_enviar = Button(root, text="Enviar", command=enviar_dados)
    botao_enviar.pack(pady=10)

    # Área de texto para exibir os resultados
    texto_resultado = Text(root, height=10, width=60)
    texto_resultado.pack(pady=10)

    # Barra de rolagem para a área de texto
    scrollbar = Scrollbar(root, command=texto_resultado.yview)
    texto_resultado.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    # Frame para exibir o gráfico
    frame_grafico = Frame(root)
    frame_grafico.pack(pady=10)

    # Inicia o loop da interface gráfica
    root.mainloop()

# Executa a interface gráfica
if __name__ == "__main__":
    exibir_interface()