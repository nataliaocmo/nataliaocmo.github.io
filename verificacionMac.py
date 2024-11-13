import subprocess

comando = "getmac"  # En Windows, usa "dir". En Linux o macOS, usa "ls"
macPc = subprocess.run(comando, shell=True, text=True, capture_output=True)

# Obtener el resultado del comando
salida = macPc.stdout

i=0

# Procesar la salida para extraer solo la primera dirección MAC
direccion_mac = None
for linea in salida.splitlines():
    partes = linea.split()
    if i==3:
        direccion_mac = partes[0].strip()
        break
    i=i+1

# Imprimir la dirección MAC
print(direccion_mac)

