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
    Binariza a imagem destacando apenas as regiões mais claras (arroz).
    
    Args:
        img: Imagem original em escala BGR
    
    Returns:
        Imagem binarizada (preto e branco)
    """
    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suaviza ruído sem destruir tanto as bordas dos grãos
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Mantém somente os pixels mais claros (arroz) via limiar automático (Otsu)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Limpa pequenos pontos e fecha falhas em regiões do arroz
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary

def count(img):
    """
    Método de contagem de grãos.
    
    Args:
        img: Imagem original (BGR)
    
    Returns:
        Tupla com (quantidade_de_grãos, imagem_binarizada)
    """
    # Binariza a imagem
    binary = binarize(img)
    
    # Encontra contornos e conta grãos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grain_count = sum(1 for c in contours if cv2.contourArea(c) >= MIN_AREA)
    
    return grain_count, binary


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
            
            # Conta os grãos e recebe a imagem binarizada
            contado, binary = count(img)
            
            # Armazena os dados das imagens
            images_data.append({
                "arquivo": filename,
                "original": img,
                "binary": binary
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
                    show_image_pipeline(
                        images_data[idx]['arquivo'],
                        images_data[idx]['original'],
                        images_data[idx]['binary']
                    )
                else:
                    print(f"❌ Por favor, insira um número entre 0 e {len(images_data)-1}, 'table' ou 'exit'")
        except ValueError:
            print("❌ Por favor, insira um número, 'table' ou 'exit'")

if __name__ == "__main__":
    main()
