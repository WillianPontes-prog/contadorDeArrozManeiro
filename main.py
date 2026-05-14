"""
Estimador de grãos de arroz em imagens BMP.

Infos legais 👍

1. 🤥 Cada emoji foi posicionado com cuidado por mim, isso não é coisa de IA,
    achei bom deixar claro isso. Estou tentando comentar esse trabalho de forma 
    um pouco mais organizada e engraçadinha

2. 📚 Pra adicionar mais imagens nos testes, basta editar a lista IMAGES
    no topo do arquivo.

    Formato:
        ("nome.bmp", quantidade_real)

    Exemplo:
        ("128.bmp", 128)

3. 🔬 O algoritmo funciona como uma pipeline de processamento de imagem,
    onde vários tratamentos são aplicados em sequência.

    Etapas principais:

    ✓ Detecção de bordas usando gradiente (Sobel)
    ✓ Remoção de ruídos usando morfologia matemática
    ✓ Criação de máscaras por regiões conectadas
    ✓ Binarização local adaptativa por blob
    ✓ Análise estatística das áreas detectadas
    ✓ Estimativa da quantidade de grãos por proporção de área

4. 🎯 A binarização NÃO usa apenas threshold simples global.

    Primeiro detectamos mudanças abruptas de intensidade
    (bordas do arroz) usando gradientes.

    Isso ajuda a eliminar:

    ✗ Reflexos suaves
    ✗ Degradês de iluminação
    ✗ Variações leves de claridade
    ✗ Pequenos ruídos do fundo

5. 📏 A contagem final é feita usando estatística das áreas.

    Como alguns grãos podem estar encostados, o algoritmo estima
    a área típica de um único grão e usa isso para decidir quando
    um blob provavelmente representa 2 ou mais grãos conectados.

6. 🧠 A estimativa da área típica é robusta contra blobs grandes.

    O algoritmo descarta iterativamente regiões muito grandes,
    evitando que grupos de grãos conectados distorçam a média.

"""

from pathlib import Path
import cv2
import numpy as np


# Parâmetros
MIN_AREA = 30

# Lista de imagens para testes
IMAGES = [
    ("60.bmp",  60),
    ("82.bmp",  82),
    ("114.bmp", 114),
    ("150.bmp", 150),
    ("205.bmp", 205),
]


# ========================================================================================== #
# Pipeline de processamento 🔬


def pipeline(img) -> tuple:
    """
    Pipeline completa de detecção e contagem de grãos de arroz.

    Etapas:
        1. Detecção de bordas por gradiente (Sobel)
        2. Erosão para remover ruído nas bordas
        3. Dilatação para criar máscara por região
        4. Binarização local por blob
        5. Estimativa da área típica de 1 grão
        6. Contagem por divisão de área

    Args:
        img: Imagem original

    Returns:
        Tupla (contagem, binary, edges, eroded, mask)
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges  = detect_edges(img)
    eroded = erode_edges(edges)
    mask   = dilate_mask(eroded)
    binary = local_threshold_by_blob(gray, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    areas = [
        int(stats[lbl, cv2.CC_STAT_AREA])
        for lbl in range(1, num_labels)
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_AREA
    ]

    if not areas:
        return 0, binary, edges, eroded, mask

    single_grain_area = estimate_single_grain_area(areas)

    # cada blob contribui com round(area / area_1_grao) grãos — mínimo 1
    total = sum(
        max(1, round(area / single_grain_area))
        for area in areas
    )

    return total, binary, edges, eroded, mask


# ========================================================================================== #
# Etapas da pipeline 🔧


def detect_edges(img) -> np.ndarray:
    """
    Detecta bordas por gradiente de intensidade (Sobel).

    Mudanças abruptas (borda do grão vs fundo) geram alta magnitude.
    Mudanças suaves (degradê de luz, reflexo) geram baixa magnitude e
    são eliminadas pelo threshold.

    Args:
        img: Imagem original (BGR)

    Returns:
        Imagem binarizada com as bordas detectadas
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    sobelx    = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    _, binary = cv2.threshold(magnitude, 65, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    return binary


def erode_edges(edges) -> np.ndarray:
    """
    Erode as bordas detectadas para remover ruídos pequenos e fragmentos isolados.

    Args:
        edges: Imagem binarizada com bordas detectadas

    Returns:
        Imagem erodida
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.erode(edges, kernel, iterations=1)


def dilate_mask(edges) -> np.ndarray:
    """
    Dilata as bordas erodidas para criar regiões fechadas (máscara por blob).

    Cada blob resultante engloba aproximadamente um grão ou grupo de grãos
    colados, e será usado como região de interesse para a binarização local.

    Args:
        edges: Imagem erodida com bordas

    Returns:
        Máscara com blobs dilatados e fechados
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.dilate(edges, kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def local_threshold_by_blob(gray, mask) -> np.ndarray:
    """
    Binariza cada blob da máscara individualmente usando um limiar local.

    O limiar de cada blob é calculado como média + 0.5 × desvio padrão
    dos pixels originais dentro daquela região. Isso adapta o threshold
    a variações locais de brilho (ex: grãos mais escuros num canto iluminado).

    Args:
        gray: Imagem em escala de cinza original
        mask: Máscara binária com os blobs

    Returns:
        Imagem binária final
    """
    result       = np.zeros_like(gray)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < MIN_AREA:
            continue

        blob_mask         = cv2.compare(labels, lbl, cv2.CMP_EQ)
        mean, stddev      = cv2.meanStdDev(gray, mask=blob_mask)
        threshold         = float(np.clip(mean[0][0] + 0.5 * stddev[0][0], 0, 255))

        _, local_binary   = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        local_binary      = cv2.bitwise_and(local_binary, blob_mask)
        result            = cv2.bitwise_or(result, local_binary)

    return result


def estimate_single_grain_area(areas: list[float]) -> float:
    """
    Estima a área de um único grão de forma iterativa.

    Parte da mediana de todos os blobs e descarta progressivamente os blobs
    maiores que 1.5x a estimativa atual, convergindo para o tamanho típico
    de um grão isolado. Usa o percentil 45 para não superestimar a típica
    quando há muitos blobs grandes (grupos de grãos colados).

    Args:
        areas: Lista de áreas dos blobs detectados (em pixels)

    Returns:
        Área estimada de um único grão (float, mínimo 1.0)
    """
    arr      = np.array(areas, dtype=np.float32)
    estimate = float(np.median(arr))

    for _ in range(10):
        single_grain_candidates = arr[arr <= estimate * 1.5]

        if len(single_grain_candidates) == 0:
            break

        new_estimate = float(np.percentile(single_grain_candidates, 45))

        if abs(new_estimate - estimate) < 1.0:
            break

        estimate = new_estimate

    return max(estimate, 1.0)


# ========================================================================================== #
# Visualização 👁️


def show_pipeline_stages(filename: str, original, edges, eroded, mask, binary):
    """
    Exibe todos os estágios da pipeline em janelas separadas.

    Args:
        filename: Nome do arquivo
        original: Imagem original (BGR)
        edges:    Bordas detectadas pelo Sobel
        eroded:   Bordas após erosão
        mask:     Máscara dilatada por blob
        binary:   Binarização local final
    """
    cv2.imshow(f"01 - Original      - {filename}", original)
    cv2.imshow(f"02 - Bordas        - {filename}", cv2.cvtColor(edges,  cv2.COLOR_GRAY2BGR))
    cv2.imshow(f"03 - Bordas Erodidas - {filename}", cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR))
    cv2.imshow(f"04 - Mascara       - {filename}", cv2.cvtColor(mask,   cv2.COLOR_GRAY2BGR))
    cv2.imshow(f"05 - Binarizada    - {filename}", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

    print(f"\n🎨 Pipeline completa — {filename}:")
    print(f"  01 Original       : imagem bruta")
    print(f"  02 Bordas         : gradiente Sobel binarizado")
    print(f"  03 Bordas Erodidas: ruídos removidos por erosão")
    print(f"  04 Mascara        : blobs dilatados para análise local")
    print(f"  05 Binarizada     : threshold local por blob (média + desvio padrão)")
    print(f"Pressione qualquer tecla para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_gradient_stages(filename: str, original):
    """
    Exibe as etapas intermediárias da detecção de bordas por gradiente.

    Args:
        filename: Nome do arquivo
        original: Imagem original (BGR)
    """
    gray    = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    sx      = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy      = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag     = cv2.magnitude(sx, sy)
    mag     = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    _, b    = cv2.threshold(mag, 65, 255, cv2.THRESH_BINARY)
    k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    b       = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=2)
    b       = cv2.morphologyEx(b, cv2.MORPH_OPEN,  k, iterations=1)

    cv2.imshow(f"01 - Original          - {filename}", original)
    cv2.imshow(f"02 - Escala de Cinza   - {filename}", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    cv2.imshow(f"03 - Mapa de Gradientes- {filename}", cv2.cvtColor(mag,  cv2.COLOR_GRAY2BGR))
    cv2.imshow(f"04 - Bordas Detectadas - {filename}", cv2.cvtColor(b,    cv2.COLOR_GRAY2BGR))

    print(f"\n🎯 Detecção de gradientes — {filename}:")
    print(f"  03 Mapa de Gradientes:")
    print(f"    ✓ Branco = mudanças abruptas (borda do grão)")
    print(f"    ✗ Preto  = mudanças suaves (reflexos, degradê — eliminados)")
    print(f"Pressione qualquer tecla para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_final_binary(filename: str, original, binary):
    """
    Exibe apenas a imagem original e a binarização final lado a lado.

    Args:
        filename: Nome do arquivo
        original: Imagem original (BGR)
        binary:   Imagem binarizada final
    """
    cv2.imshow(f"Original  - {filename}", original)
    cv2.imshow(f"Binarizada - {filename}", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

    print(f"\n📸 {filename} — pressione qualquer tecla para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ========================================================================================== #
# Utilitários 🛠️
# Métodos abaixo gerados por IA, pois são apenas para abrir arquivo e comparar resultados ☝️


def load_image(filename: str) -> np.ndarray:
    """
    Carrega uma imagem BMP do disco.

    Args:
        filename: Caminho do arquivo BMP

    Returns:
        Imagem original (BGR)

    Raises:
        ValueError: Se o arquivo não puder ser lido
    """
    img = cv2.imread(str(Path(filename)))
    if img is None:
        raise ValueError(f"{filename}: não foi possível ler a imagem")
    return img


def build_result(estimado: int, real: int) -> dict:
    """
    Monta o dicionário de resultado de uma estimativa.

    Args:
        estimado: Contagem estimada
        real:     Contagem real esperada

    Returns:
        Dict com estimado, real, erro e percentual
    """
    erro = estimado - real
    return {
        "estimado":   estimado,
        "real":       real,
        "erro":       erro,
        "percentual": (erro / real * 100) if real > 0 else 0.0,
    }


def print_results_table(resultados: list):
    """
    Imprime a tabela de resultados formatada.

    Args:
        resultados: Lista de dicts retornados por build_result (com chave 'arquivo')
    """
    print("\n" + "=" * 80)
    print(f"{'Arquivo':<15} {'Estimado':>12} {'Real':>12} {'Erro':>12} {'Percentual':>15}")
    print("=" * 80)

    total_error = 0
    for res in resultados:
        print(
            f"{res['arquivo']:<15} {res['estimado']:>12d} {res['real']:>12d} "
            f"{res['erro']:>+12d} {res['percentual']:>+14.1f}%"
        )
        total_error += abs(res['erro'])

    print("=" * 80)
    print(f"{'Erro absoluto médio':<41} {total_error / len(resultados):>11.1f}")
    print("=" * 80)


# ========================================================================================== #
# Main 🚀


def main():
    print("Processando imagens...\n")

    resultados  = []
    images_data = []

    for filename, real in IMAGES:
        try:
            img = load_image(filename)
            contado, binary, edges, eroded, mask = pipeline(img)

            images_data.append({
                "arquivo":  filename,
                "original": img,
                "binary":   binary,
                "edges":    edges,
                "eroded":   eroded,
                "mask":     mask,
            })

            resultado             = build_result(contado, real)
            resultado['arquivo']  = filename
            resultados.append(resultado)

        except Exception as exc:
            print(f"Erro processando {filename}: {exc}")

    print_results_table(resultados)

    print("\n📁 Imagens disponíveis para visualização:")
    for i, data in enumerate(images_data):
        print(f"  {i}: {data['arquivo']}")
    print("  table : mostrar tabela novamente")
    print("  exit  : sair")

    while True:
        try:
            choice = input(f"\n🔍 Qual imagem? (0-{len(images_data)-1}, 'table' ou 'exit'): ")

            if choice.lower() == "exit":
                print("👋 Saindo...\n")
                break

            elif choice.lower() == "table":
                print_results_table(resultados)

            else:
                idx = int(choice)
                if not (0 <= idx < len(images_data)):
                    print(f"❌ Número entre 0 e {len(images_data)-1}")
                    continue

                data = images_data[idx]
                print("\n📊 Opções de visualização:")
                print("  1: Resultado final")
                print("  2: Etapas do gradiente")
                print("  3: Pipeline completa")

                view = input("Escolha (1, 2 ou 3): ").strip()

                if view == "1":
                    show_final_binary(data['arquivo'], data['original'], data['binary'])
                elif view == "2":
                    show_gradient_stages(data['arquivo'], data['original'])
                elif view == "3":
                    show_pipeline_stages(
                        data['arquivo'], data['original'],
                        data['edges'],   data['eroded'],
                        data['mask'],    data['binary'],
                    )
                else:
                    print("❌ Use 1, 2 ou 3.")

        except ValueError:
            print("❌ Insira um número, 'table' ou 'exit'")


if __name__ == "__main__":
    main()