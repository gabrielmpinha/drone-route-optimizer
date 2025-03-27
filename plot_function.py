import matplotlib.pyplot as plt
import numpy as np

def plot_results(res, locacoes_cidades, initial_pos, nomes_cidades):
    tempo = res.F[:, 0]
    energia = res.F[:, 1]
    velocidade = res.X[:, -1] / 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Segundo gráfico: Rota do Drone para até 3 soluções
    sorted_indices = np.lexsort((energia, velocidade))
    selected_indices = sorted_indices[::len(sorted_indices) // 3][:3]

    # Primeiro gráfico: Tempo x Energia x velocidade
    edgecolors = ['blue' if i in selected_indices else 'red' for i in range(len(tempo))]
    scatter = ax1.scatter(tempo, energia, c=velocidade, cmap='viridis', edgecolor=edgecolors)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Velocidade')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Energia')
    ax1.set_title('Tempo x Energia x Velocidade')

    annot = ax1.annotate("", xy=(0, 0), xytext=(20, 20),
                         textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="w"),
                         arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        sol_index = ind["ind"][0]
        text = f"Solução: {sol_index}\nVelocidade: {velocidade[sol_index]:.2f}\nTempo: {tempo[sol_index]:.2f}\nEnergia: {energia[sol_index]:.2f}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('yellow')
        annot.get_bbox_patch().set_alpha(0.6)

    def on_click(event):
        vis = annot.get_visible()
        if event.inaxes == ax1:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    cmap = scatter.get_cmap()
    norm = scatter.norm

    lines = []

    for i in selected_indices:
        ordem_indices = res.X[i, :-1]  # Exclui a última coluna (velocidade)
        ordem_cidades = [nomes_cidades[j] for j in ordem_indices]

        # Obter as coordenadas X e Y das cidades a partir da lista de objetos Pacote
        x_coords = [initial_pos[0]] + [next(cidade.x for cidade in locacoes_cidades if cidade.nome == nome) for nome in ordem_cidades] + [initial_pos[0]]
        y_coords = [initial_pos[1]] + [next(cidade.y for cidade in locacoes_cidades if cidade.nome == nome) for nome in ordem_cidades] + [initial_pos[1]]

        color = cmap(norm(velocidade[i]))
        line, = ax2.plot(x_coords, y_coords, marker='o', linestyle='-', color=color, label=f'Solução {i}')
        lines.append(line)
        for j, cidade in enumerate(['Inicial'] + ordem_cidades):
            if j == 1:
                ax2.text(x_coords[j], y_coords[j], cidade, fontsize=12, ha='right', color='red')
            else:
                ax2.text(x_coords[j], y_coords[j], cidade, fontsize=12, ha='right')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Rota do Drone')
    ax2.grid(True)
    legend = ax2.legend()

    def on_pick(event):
        legend_item = event.artist
        orig_line = lines[legend.get_texts().index(legend_item)]
        visible = not orig_line.get_visible()
        orig_line.set_visible(visible)

        if not visible:
            legend_item.set_text(''.join([char + '\u0336' for char in legend_item.get_text()]))
        else:
            legend_item.set_text(legend_item.get_text().replace('\u0336', ''))
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)

    for legend_item in legend.get_texts():
        legend_item.set_picker(True)

    plt.tight_layout()

    # Retorna o objeto Figure em vez de exibir o gráfico
    return fig