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
            
            # Conta os grãos e recebe a imagem binarizada
            contado, binary = count(img)
            
            # Extrai dados para visualização de gradientes
            gray, magnitude, _ = visualize_gradient_detection(img)
            
            # Armazena os dados das imagens
            images_data.append({
                "arquivo": filename,
                "original": img,
                "binary": binary,
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
                    
                    view_choice = input("Escolha (1 ou 2): ").strip()
                    
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
                    else:
                        print("❌ Escolha inválida. Use 1 ou 2.")
                else:
                    print(f"❌ Por favor, insira um número entre 0 e {len(images_data)-1}, 'table' ou 'exit'")
        except ValueError:
            print("❌ Por favor, insira um número, 'table' ou 'exit'")


if __name__ == "__main__":
    main()
