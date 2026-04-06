#!/usr/bin/env bash
# =============================================================================
# setup-new-distro.sh
# Instala herramientas en Ubuntu 24.04 para gimme-a-chance.
# La migracion de archivos (.ssh, .claude, etc.) ya la hizo el script de
# PowerShell (setup-ubuntu-2404.ps1). Este script solo instala software.
# Correr DENTRO de la nueva distro: bash /tmp/setup-new-distro.sh
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} gimme-a-chance: New Distro Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ===========================================================================
# VERIFICAR: archivos migrados por el script de PowerShell
# ===========================================================================
echo -e "${YELLOW}[Check] Verificando archivos migrados por setup-ubuntu-2404.ps1...${NC}"

check_item() {
    if [ -e "$1" ]; then
        echo -e "  ${GREEN}✓${NC} $1"
    else
        echo -e "  ${RED}✗${NC} $1 — NO ENCONTRADO. Correr setup-ubuntu-2404.ps1 primero?"
    fi
}

check_item "$HOME/.ssh"
check_item "$HOME/.gitconfig"
check_item "$HOME/.claude"
check_item "$HOME/gimme-a-chance"
echo ""

# Fijar permisos SSH si existe
if [ -d "$HOME/.ssh" ]; then
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/* 2>/dev/null || true
fi

# ===========================================================================
# FASE 4: Node.js (nvm + Node 22 LTS)
# ===========================================================================
echo -e "${YELLOW}[Fase 4] Instalando nvm + Node 22 LTS...${NC}"

if command -v node &>/dev/null && node --version | grep -q "v22"; then
    echo -e "  ${GREEN}✓${NC} Node 22 ya instalado: $(node --version)"
else
    export NVM_DIR="$HOME/.nvm"
    if [ ! -s "$NVM_DIR/nvm.sh" ]; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    fi

    # Cargar nvm
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

    nvm install 22
    nvm alias default 22

    echo -e "  ${GREEN}✓${NC} Node $(node --version) instalado, npm $(npm --version)"
fi
echo ""

# ===========================================================================
# FASE 5: Claude Code
# ===========================================================================
echo -e "${YELLOW}[Fase 5] Instalando Claude Code...${NC}"

# Recargar nvm por si acaso
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

if command -v claude &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Claude Code ya instalado: $(claude --version 2>&1 | head -1)"
else
    npm install -g @anthropic-ai/claude-code
    echo -e "  ${GREEN}✓${NC} Claude Code instalado: $(claude --version 2>&1 | head -1)"
fi
echo ""

# ===========================================================================
# FASE 6: Rust
# ===========================================================================
echo -e "${YELLOW}[Fase 6] Instalando Rust...${NC}"

if command -v rustc &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Rust ya instalado: $(rustc --version)"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
    echo -e "  ${GREEN}✓${NC} Rust instalado: $(rustc --version)"
fi
echo ""

# ===========================================================================
# FASE 7: Dependencias del sistema (Tauri v2 + audio)
# ===========================================================================
echo -e "${YELLOW}[Fase 7] Instalando dependencias del sistema para Tauri v2 + audio...${NC}"
echo -e "  (requiere sudo — te va a pedir password)"
echo ""

sudo apt-get update -qq

sudo apt-get install -y -qq \
    libwebkit2gtk-4.1-dev \
    libappindicator3-dev \
    librsvg2-dev \
    patchelf \
    libssl-dev \
    libgtk-3-dev \
    libsoup-3.0-dev \
    libjavascriptcoregtk-4.1-dev \
    libasound2-dev \
    build-essential \
    pkg-config \
    cmake \
    libglib2.0-dev \
    clang

echo -e "  ${GREEN}✓${NC} Dependencias del sistema instaladas"
echo ""

# ===========================================================================
# FASE 8: Verificar que gimme-a-chance compila
# ===========================================================================
echo -e "${YELLOW}[Fase 8] Verificando que gimme-a-chance compila...${NC}"

# Asegurarse de que cargo está en PATH
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

if [ -d "$HOME/gimme-a-chance/src-tauri" ]; then
    cd "$HOME/gimme-a-chance/src-tauri"
    echo "  Corriendo cargo check (esto puede tardar unos minutos la primera vez)..."
    if cargo check 2>&1; then
        echo -e "  ${GREEN}✓${NC} Compilacion exitosa!"
    else
        echo -e "  ${RED}✗${NC} Hubo errores de compilacion. Revisar arriba."
    fi
else
    echo -e "  ${YELLOW}⚠${NC} ~/gimme-a-chance/ no encontrado. Saltando verificacion."
fi
echo ""

# ===========================================================================
# FASE 8b: Descargar modelo Whisper
# ===========================================================================
echo -e "${YELLOW}[Fase 8b] Descargando modelo Whisper (base.en)...${NC}"

MODEL_DIR="$HOME/.local/share/gimme-a-chance/models"
MODEL_FILE="$MODEL_DIR/ggml-base.en.bin"

if [ -f "$MODEL_FILE" ]; then
    echo -e "  ${GREEN}✓${NC} Modelo ya existe en $MODEL_FILE"
else
    mkdir -p "$MODEL_DIR"
    echo "  Descargando ggml-base.en.bin (~148MB)..."
    curl -L -o "$MODEL_FILE" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    echo -e "  ${GREEN}✓${NC} Modelo descargado"
fi
echo ""

# ===========================================================================
# DONE
# ===========================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Setup completado!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Resumen:"
echo -e "  Node:       $(node --version 2>/dev/null || echo 'no encontrado')"
echo -e "  npm:        $(npm --version 2>/dev/null || echo 'no encontrado')"
echo -e "  Claude:     $(claude --version 2>&1 | head -1 || echo 'no encontrado')"
echo -e "  Rust:       $(rustc --version 2>/dev/null || echo 'no encontrado')"
echo -e "  Cargo:      $(cargo --version 2>/dev/null || echo 'no encontrado')"
echo ""
echo -e "Proximo paso:"
echo -e "  ${CYAN}cd ~/gimme-a-chance && claude${NC}"
echo -e "  Y seguimos construyendo gimme-a-chance!"
echo ""
