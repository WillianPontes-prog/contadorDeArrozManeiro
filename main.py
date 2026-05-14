"""
Estimador de grãos de arroz em imagens BMP.

Infos legais 👍

1. 🤥 Cada emoji foi posicionado com cuidado por mim, isso não é coisa de IA, bom deixar claro, 
    estou pondo um pouco mais de trabalho nesse trabalho 4 em especifico, então vou tentar comentar
    um pouco mais bonitinho (sim esse texto todo é pra eu usar emojis sem ficar estranho).

2. 📚 Para adicionar mais imagens nos testes, basta editar a lista 'IMAGES' no topo do arquivo.
    Cada linha deve ter o formato: ("nome_do_arquivo.bmp", valor_real_esperado)
    Exemplo: ("128.bmp", 128) - será processada e o resultado comparado com 128 grãos esperados.

3. 
        
"""

from pathlib import Path
import cv2
import numpy as np


# Define de parâmetros
BLOCKSIZE = 11
OFFSET = -19
MIN_AREA = 30

# Debug - mostrar imagens intermediárias durante o processamento
SHOW_DEBUG = False

# Lista de imagens para testes
IMAGES = [
    ("60.bmp", 60),
    ("82.bmp", 82),
    ("114.bmp", 114),
    ("150.bmp", 150),
    ("205.bmp", 205),
]


def binarize(img):
    """
    Binariza a imagem detectando mudanças ABRUPTAS de intensidade.
    Isso elimina luzes com degradê suave e detecta bordas do arroz vs fundo.
    
    Args:
        img: Imagem original em escala BGR
    
    Returns:
        Imagem binarizada baseada em gradientes (contornos)
    """
    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suaviza ruído sem destruir as bordas do arroz
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    
    # Calcula os gradientes usando Sobel (derivadas em X e Y)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcula a magnitude do gradiente (força da mudança de intensidade)
    magnitude = cv2.magnitude(sobelx, sobely)
    
    # Normaliza para 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # Aplica threshold para detectar apenas mudanças significativas (bordas fortes)
    # Valores altos = mudanças abruptas (arroz vs fundo)
    # Valores baixos = mudanças suaves (luzes com degradê)
    # Threshold de 120: apenas as mudanças mais abruptas são consideradas bordas
    _, binary = cv2.threshold(magnitude, 65, 255, cv2.THRESH_BINARY)
    
    # Aplica erosão e dilatação para limpar ruído e conectar regiões
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary

def count(img):
    
    """
    Conta os grãos de arroz baseado na área dos blobs brancos.

    Estratégia:
    - Cada componente conectado branco é analisado
    - Descobre a área típica de 1 grão
    - Componentes maiores contam como 2, 3, etc.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # BINARIZAÇÃO ORIGINAL
    # ============================================================

    edges = binarize(img)

    eroded = erode_borders(edges)

    mask = fill_borders(eroded)

    binary = local_binarize_by_mask(gray, mask)

    # limpeza final
    kernel_final = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel_final,
        iterations=1
    )

    # ============================================================
    # COMPONENTES CONECTADOS
    # ============================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    areas = []

    for lbl in range(1, num_labels):

        area = int(stats[lbl, cv2.CC_STAT_AREA])

        if area >= MIN_AREA:
            areas.append(area)

    if len(areas) == 0:
        return 0, binary, edges, eroded, mask

    # ============================================================
    # ÁREA TÍPICA DE 1 ARROZ
    # ============================================================

    areas_arr = np.array(areas, dtype=np.float32)

    # usa os menores blobs como referência
    percentile_60 = np.percentile(areas_arr, 60)

    small_areas = areas_arr[
        areas_arr <= percentile_60
    ]

    if len(small_areas) > 0:
        typical_area = np.median(small_areas)
    else:
        typical_area = np.median(areas_arr)

    typical_area = max(typical_area, 1.0)

    # ============================================================
    # CONTAGEM
    # ============================================================

    total = 0

    for lbl in range(1, num_labels):

        area = int(stats[lbl, cv2.CC_STAT_AREA])

        if area < MIN_AREA:
            continue

        # estima quantos arrozes existem nesse blob
        estimate = round(area / typical_area)

        # mínimo 1
        estimate = max(1, estimate)

        total += estimate

    return total, binary, edges, eroded, mask
def fill_borders(edges):
    """
    Dilata as bordas para criar uma máscara por blob.

    Args:
        edges: Imagem binarizada com as bordas (contornos)

    Returns:
        Máscara dilatada com os blobs unidos o suficiente para a análise local
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    filled = cv2.dilate(edges, kernel, iterations=1)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
    return filled


def erode_borders(edges):
    """
    Aplica erosão nas bordas para limpar ruídos e peças pequenas.

    Args:
        edges: Imagem binarizada com as bordas (contornos)

    Returns:
        Imagem erodida (ruídos removidos)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(edges, kernel, iterations=1)
    return eroded


def local_binarize_by_mask(gray, mask):
    """
    Binariza localmente cada blob branco da máscara usando a imagem gray original.
    O limiar de cada blob é calculado como media + desvio padrao.

    Args:
        gray: Imagem em escala de cinza original
        mask: Máscara binária com os blobs brancos separados

    Returns:
        Imagem binária final, calculada blob a blob
    """
    result = np.zeros_like(gray)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, labels_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue

        component_mask = cv2.compare(labels, label, cv2.CMP_EQ)
        mean, stddev = cv2.meanStdDev(gray, mask=component_mask)
        threshold = float(mean[0][0] + (0.5 * stddev[0][0]))
        threshold = max(0.0, min(255.0, threshold))

        _, local_binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        local_binary = cv2.bitwise_and(local_binary, component_mask)
        result = cv2.bitwise_or(result, local_binary)

    return result


def split_large_components_watershed(img_bgr, binary, labels, stats, typical_area, rel_thresh=0.18):
    """
    Tenta separar componentes muito grandes usando watershed com marcas baseadas em picos

    Args:
        img_bgr: imagem BGR original (usada como entrada para watershed)
        binary: imagem binária atual (0/255)
        labels: rótulos dos componentes atuais
        stats: estatísticas dos componentes atuais (output do connectedComponentsWithStats)
        typical_area: área típica de um grão (float)
        rel_thresh: fração do máximo da distance transform para considerar picos

    Returns:
        binary_mod: nova imagem binária onde grandes componentes podem ter sido separados
    """
    binary_mod = binary.copy()
    h, w = binary.shape[:2]

    # limite de área para considerar um componente 'grande'
    large_factor = 1.8
    for lbl in range(1, stats.shape[0]):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area <= int(large_factor * typical_area):
            continue

        # máscara do componente
        comp_mask = (labels == lbl).astype('uint8') * 255

        # distancia dentro do componente
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            continue

        # detecta picos locais na distância (marcadores para watershed)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dil = cv2.dilate(dist, k)
        peaks = (dist == dil) & (dist >= (dist.max() * rel_thresh))
        peaks_u8 = (peaks.astype('uint8') * 255)
        peaks_u8 = cv2.morphologyEx(peaks_u8, cv2.MORPH_OPEN, k, iterations=1)

        n_markers, markers = cv2.connectedComponents(peaks_u8)
        if n_markers <= 1:
            # não encontrou sementes suficientes para separar
            continue

        # prepara marcador para watershed: rotulos >0
        marker_img = np.zeros((h, w), dtype=np.int32)
        # atribui rótulos dos marcadores (1..n_markers-1)
        marker_img[peaks_u8 > 0] = markers[peaks_u8 > 0]

        # imagem de entrada para watershed: região da componente recortada sobre o BGR original
        region = cv2.bitwise_and(img_bgr, img_bgr, mask=comp_mask)

        # watershed modifica marker_img in-place
        try:
            cv2.watershed(region, marker_img)
        except Exception:
            # fallback: ignora se watershed falhar
            continue

        # markers > 1 correspondem a regiões separadas (watershed usa -1 para borda)
        unique_markers = np.unique(marker_img)
        # ignora -1 e 0
        new_comp = np.zeros_like(comp_mask)
        for m in unique_markers:
            if m <= 0:
                continue
            mask_m = (marker_img == m).astype('uint8') * 255
            # adiciona região separada ao novo componente
            new_comp = cv2.bitwise_or(new_comp, mask_m)

        # substitui a área do componente original pela nova separação
        # remove a região antiga e cola a nova
        binary_mod[comp_mask > 0] = 0
        binary_mod[new_comp > 0] = 255

    return binary_mod


def visualize_gradient_detection(img):
    """
    Visualiza o processo de detecção de gradientes (contornos com mudanças abruptas).
    
    Args:
        img: Imagem original em escala BGR
    
    Returns:
        Tupla com (gray, magnitude, binary)
    """
    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suaviza ruído
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    
    # Calcula os gradientes usando Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcula a magnitude do gradiente
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # Binariza baseado nos gradientes (threshold 120: mudanças muito abruptas)
    _, binary = cv2.threshold(magnitude, 120, 255, cv2.THRESH_BINARY)
    
    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return gray, magnitude, binary



# ========================================================================================== #
# Pra deixar explicito dessa vez ☝️
# Metodos abaixo gerados por IA, pois são apenas para abrir arquivo e comparar resultados

def print_results_table(resultados):
    """
    Imprime a tabela de resultados.
    
    Args:
        resultados: Lista com os resultados de cada imagem
    """
    print("\n" + "="*80)
    print(f"{'Arquivo':<15} {'Estimado':>12} {'Real':>12} {'Erro':>12} {'Percentual':>15}")
    print("="*80)
    
    total_error = 0
    for res in resultados:
        print(f"{res['arquivo']:<15} {res['estimado']:>12d} {res['real']:>12d} {res['erro']:>+12d} {res['percentual']:>+14.1f}%")
        total_error += abs(res['erro'])
    
    print("="*80)
    print(f"{'Erro absoluto médio':<41} {total_error / len(resultados):>11.1f}")
    print("="*80)


def show_image_pipeline(filename: str, original, binary):
    """
    Exibe todos os estágios da pipeline de processamento de uma imagem em janelas separadas.
    
    Args:
        filename: Nome do arquivo
        original: Imagem original (BGR)
        binary: Imagem binarizada
    """
    # Exibe a imagem original
    cv2.imshow(f"Original - {filename}", original)
    
    # Converte a imagem binarizada para BGR para exibição
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f"Binarizada - {filename}", binary_bgr)
    
    print(f"\n📸 Mostrando {filename}:")
    print(f"  - Janela 'Original': imagem bruta")
    print(f"  - Janela 'Binarizada': resultado do processamento")
    print(f"Pressione qualquer tecla para fechar as janelas e voltar ao menu.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_filled_borders(filename: str, original, edges, eroded, mask, binary):
    """
    Exibe a sequência completa: bordas detectadas -> erodidas -> máscara -> binarização local.
    
    Args:
        filename: Nome do arquivo
        original: Imagem original (BGR)
        edges: Imagem com as bordas detectadas
        eroded: Imagem após erosão (ruídos removidos)
        mask: Máscara dilatada usada na binarização local
        binary: Binarização local final
    """
    cv2.imshow(f"01 - Original - {filename}", original)

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f"02 - Bordas Detectadas - {filename}", edges_bgr)

    eroded_bgr = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f"03 - Bordas Erodidas - {filename}", eroded_bgr)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f"04 - Mascara Diluida - {filename}", mask_bgr)

    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f"05 - Binarizacao Local - {filename}", binary_bgr)

    print(f"\n🎨 Binarizacao Local por Blob - {filename}:")
    print(f"  - Janela '01 Original': imagem bruta")
    print(f"  - Janela '02 Bordas Detectadas': resposta inicial do detector")
    print(f"  - Janela '03 Bordas Erodidas': limpa ruídos pequenos")
    print(f"  - Janela '04 Mascara Diluida': blobs unidos para analise local")
    print(f"  - Janela '05 Binarizacao Local': threshold por blob usando media + desvio padrao")
    print(f"Pressione qualquer tecla para fechar as janelas.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_gradient_pipeline(filename: str, original, gray, magnitude, binary):
    """
    Exibe o processo completo de detecção de gradientes.
    Mostra: cinza, mapa de gradientes, e resultado binarizado.
    
    Args:
        filename: Nome do arquivo
        original: Imagem original (BGR)
        gray: Imagem em escala de cinza
        magnitude: Mapa de gradientes (magnitude do vetor gradiente)
        binary: Imagem binarizada final
    """
    # Exibe a imagem original
    cv2.imshow(f"01 - Original - {filename}", original)
    
    # Exibe a escala de cinza
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f"02 - Escala de Cinza - {filename}", gray_bgr)
    
    # Exibe o mapa de gradientes (forças de mudança)
    magnitude_bgr = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f"03 - Mapa de Gradientes - {filename}", magnitude_bgr)
    
    # Exibe a imagem binarizada
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.imshow(f"04 - Binarizada (Contornos) - {filename}", binary_bgr)
    
    print(f"\n🎯 Detecção de Gradientes - {filename}:")
    print(f"  - Janela '01 Original': imagem bruta")
    print(f"  - Janela '02 Escala de Cinza': conversão inicial")
    print(f"  - Janela '03 Mapa de Gradientes': mostra força das mudanças de intensidade")
    print(f"    ✓ Branco = mudanças abruptas (bordas do arroz vs fundo)")
    print(f"    ✗ Preto = mudanças suaves (degradê das luzes - ELIMINADAS)")
    print(f"  - Janela '04 Binarizada': contornos finais detectados")
    print(f"Pressione qualquer tecla para fechar as janelas.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(filename: str):
    """
    Abre uma imagem BMP e retorna a imagem original.
    
    Args:
        filename: Nome do arquivo BMP
    
    Returns:
        Imagem original (BGR)
    """
    path = Path(filename)
    img = cv2.imread(str(path))
    
    if img is None:
        raise ValueError(f"{filename}: não conseguiu ler a imagem")
    
    return img

def get_result(estimado: int, real: int) -> dict:
    """
    Calcula o resultado da estimação.
    
    Args:
        estimado: Valor estimado
        real: Valor real
    
    Returns:
        Dict com estimado, real, erro e percentual
    """
    erro = estimado - real
    percentual = (erro / real * 100) if real > 0 else 0
    
    return {
        "estimado": estimado,
        "real": real,
        "erro": erro,
        "percentual": percentual
    }

def main():
    """
    Função principal que processa a lista de imagens.
    """
    print("Processando imagens...\n")
    
    resultados = []
    images_data = []  # Armazena as imagens processadas
    
    for filename, real in IMAGES:
        try:
            # Carrega a imagem
            img = load_image(filename)
            
            # Conta os grãos e recebe a imagem binarizada local final
            contado, binary, edges, eroded, mask = count(img)
            
            # Extrai dados para visualização de gradientes
            gray, magnitude, _ = visualize_gradient_detection(img)
            
            # Armazena os dados das imagens
            images_data.append({
                "arquivo": filename,
                "original": img,
                "binary": binary,
                "edges": edges,
                "eroded": eroded,
                "mask": mask,
                "gray": gray,
                "magnitude": magnitude
            })
            
            # Valida o resultado
            resultado = get_result(contado, real)
            resultado['arquivo'] = filename
            
            resultados.append(resultado)
            
        except Exception as exc:
            print(f"Erro processando {filename}: {exc}")
    
    # Imprime tabela
    print_results_table(resultados)
    
    # Pergunta qual imagem visualizar
    print("\n📁 Imagens disponíveis para visualização:")
    for i, data in enumerate(images_data):
        print(f"  {i}: {data['arquivo']}")
    print(f"  table: Mostrar tabela novamente")
    print(f"  exit: Sair")
    
    while True:
        try:
            choice = input("\n🔍 Qual imagem deseja ver? (0-{}, 'table' ou 'exit'): ".format(len(images_data)-1))
            
            if choice.lower() == "exit":
                print("👋 Saindo...\n")
                break
            elif choice.lower() == "table":
                print_results_table(resultados)
            else:
                idx = int(choice)
                if 0 <= idx < len(images_data):
                    data = images_data[idx]
                    print("\n📊 Opções de visualização:")
                    print("  1: Visualizar resultado final (binarização)")
                    print("  2: Visualizar processo de detecção de gradientes")
                    print("  3: Visualizar bordas preenchidas")
                    
                    view_choice = input("Escolha (1, 2 ou 3): ").strip()
                    
                    if view_choice == "1":
                        show_image_pipeline(
                            data['arquivo'],
                            data['original'],
                            data['binary']
                        )
                    elif view_choice == "2":
                        show_gradient_pipeline(
                            data['arquivo'],
                            data['original'],
                            data['gray'],
                            data['magnitude'],
                            data['binary']
                        )
                    elif view_choice == "3":
                        show_filled_borders(
                            data['arquivo'],
                            data['original'],
                            data['edges'],
                            data['eroded'],
                            data['mask'],
                            data['binary']
                        )
                    else:
                        print("❌ Escolha inválida. Use 1, 2 ou 3.")
                else:
                    print(f"❌ Por favor, insira um número entre 0 e {len(images_data)-1}, 'table' ou 'exit'")
        except ValueError:
            print("❌ Por favor, insira um número, 'table' ou 'exit'")


if __name__ == "__main__":
    main()
