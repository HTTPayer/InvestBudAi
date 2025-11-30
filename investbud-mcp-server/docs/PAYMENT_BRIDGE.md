# Payment Bridge - x402 Browser Payment Integration

## 🎯 ¿Qué es Payment Bridge?

Payment Bridge es un sistema que permite a tu servidor MCP solicitar pagos x402 abriendo automáticamente una ventana del navegador donde el usuario puede conectar su wallet (MetaMask o Polkadot.js), aprobar el pago, y luego continuar usando Claude Desktop.

## 🔄 Flujo de Trabajo

```
Claude Desktop → Solicita dato que requiere pago
       ↓
Servidor MCP detecta 402 Payment Required
       ↓
Payment Bridge inicia servidor local (http://localhost:3402)
       ↓
Se abre navegador con página de pago
       ↓
Usuario conecta wallet y aprueba pago
       ↓
Navegador envía firma al servidor local
       ↓
Servidor local cierra navegador y continúa
       ↓
Claude Desktop recibe los datos
```

## 🚀 Cómo Funciona

### 1. Detección Automática
Cuando el servidor MCP recibe un error `402 Payment Required` del API de InvestBud, automáticamente:

- Inicia un servidor web local en el puerto 3402-3502
- Genera un ID de sesión único
- Abre tu navegador predeterminado

### 2. Página de Pago
La página que se abre:

- Muestra todos los detalles del pago (red, destinatario, monto)
- Detecta qué tipo de wallet necesitas (MetaMask para EVM, Polkadot.js para Substrate)
- Te guía para conectar tu wallet
- Solicita firma del mensaje de pago

### 3. Aprobación
Cuando apruebes el pago en tu wallet:

- La firma se envía de vuelta al servidor local
- El navegador se cierra automáticamente
- El servidor MCP usa la firma para reintentar la petición
- Claude Desktop recibe la respuesta

## 🛠 Instalación y Uso

Ya está todo configurado en tu proyecto. Solo necesitas:

### 1. Tener MetaMask instalado
Para redes EVM (Ethereum, Base, Polygon, etc.):
- Instala [MetaMask](https://metamask.io/)
- Asegúrate de tener fondos en la red correcta

### 2. O Polkadot.js Extension
Para redes Substrate (Polkadot, Kusama):
- Instala [Polkadot.js Extension](https://polkadot.js.org/extension/)
- Configura tu cuenta

### 3. Reiniciar Claude Desktop
Después de compilar, reinicia Claude Desktop para usar la nueva versión.

## 🧪 Probar el Sistema

1. **Haz una consulta que requiera pago:**
   ```
   "Analyze this wallet: 0x364307720164378324965c27fae21242fd5807ee on base-mainnet"
   ```

2. **El navegador se abrirá automáticamente**
   - Verás la página de pago con todos los detalles

3. **Conecta tu wallet**
   - Click en "Connect Wallet & Pay"
   - Aprueba la conexión en MetaMask
   - Firma el mensaje de pago

4. **Espera confirmación**
   - La página mostrará "✅ Payment successful!"
   - Se cerrará automáticamente
   - Claude Desktop continuará con la respuesta

## 🔧 Configuración Avanzada

### Puerto del Servidor Local
Por defecto, el bridge intenta usar el puerto 3402. Si está ocupado, prueba automáticamente puertos hasta el 3502.

### Timeout de Sesión
Las sesiones de pago expiran después de 5 minutos. Si no completas el pago en ese tiempo, tendrás que reintentar.

### Redes Soportadas

**EVM (MetaMask):**
- ethereum
- base / base-mainnet / base-sepolia
- polygon / polygon-mainnet
- arbitrum / arb-mainnet
- optimism

**Substrate (Polkadot.js):**
- polkadot
- kusama
- westend

## 🐛 Troubleshooting

### El navegador no se abre
- **macOS:** Ejecuta `open http://localhost:3402/pay/[session-id]`
- **Windows:** Ejecuta `start http://localhost:3402/pay/[session-id]`
- **Linux:** Ejecuta `xdg-open http://localhost:3402/pay/[session-id]`

### Error "MetaMask not detected"
- Asegúrate de tener MetaMask instalado
- Actualiza la extensión a la última versión
- Prueba en un navegador diferente (Chrome/Brave recomendados)

### Error "No Polkadot accounts found"
- Abre la extensión Polkadot.js
- Asegúrate de tener al menos una cuenta configurada
- Autoriza el acceso cuando se solicite

### Error "Payment timeout"
- La sesión expiró (5 minutos)
- Reintenta la consulta en Claude Desktop
- Se generará una nueva sesión

### Puerto ocupado
El sistema prueba automáticamente puertos 3402-3502. Si todos están ocupados:
```bash
# Encuentra qué está usando los puertos
lsof -i :3402-3502

# Mata procesos si es necesario
kill -9 [PID]
```

## 📋 Logs y Debugging

El servidor NO escribe a stdout (protocolo MCP), pero puedes ver errores:

```bash
# Ver logs del servidor
tail -f ~/.config/claude/logs/[tu-servidor].log
```

## 🔒 Seguridad

- **Servidor local:** Solo escucha en localhost (127.0.0.1)
- **No almacena claves:** Las firmas se procesan y descartan inmediatamente
- **Sesiones únicas:** Cada pago tiene un ID único y temporal
- **Timeout automático:** Las sesiones expiran en 5 minutos

## 💡 Tips

1. **Mantén tu wallet con fondos suficientes** para pagos x402
2. **Verifica la red correcta** antes de aprobar pagos
3. **El navegador se cierra solo** cuando el pago es exitoso
4. **Puedes cancelar** con el botón "Cancel" si cambias de opinión

## 🎨 Personalización

Si quieres personalizar la página de pago, edita:
```typescript
src/services/paymentBridge.ts
// Busca la función getPaymentHTML()
```

Puedes cambiar:
- Estilos CSS
- Colores y diseño
- Mensajes de texto
- Lógica de validación

## 📞 Soporte

Si tienes problemas:
1. Verifica que MetaMask/Polkadot.js esté instalado
2. Asegúrate de tener fondos en la red correcta
3. Revisa los logs de Claude Desktop
4. Reinicia Claude Desktop
5. Recompila el proyecto: `npm run build`

---

**Nota:** Este sistema es específico para el protocolo x402 usado por InvestBud/HTTPayer. Para otros sistemas de pago, necesitarás adaptar la lógica.
