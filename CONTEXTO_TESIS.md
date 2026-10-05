# Contexto del Proyecto: Reconocimiento de Patrones y Biometría Descentralizada (SSI + ZKP)

**¡Hola futuro agente!** Si estás leyendo este archivo, es para ponerte en contexto rápidamente sobre la evolución de la arquitectura del proyecto original. Ahora está escalado y enfocado a la materia de **Reconocimiento de Patrones sobre Datos Biométricos** bajo el paradigma de Identidad Auto-Soberana (SSI) y Pruebas de Conocimiento Cero (ZKP).

## Información del Proyecto
*   **Título del Enfoque:** Modelo de Reconocimiento de Patrones y Autenticación Biométrica Preservando la Privacidad mediante Identidad Auto-Soberana (SSI).
*   **Caso de Estudio:** Autenticación biométrica descentralizada (ej. reconocimiento facial, iris o dactilar) que elimina el almacenamiento centralizado de plantillas (*templates*).
*   **Contexto de Implementación:** Materia de Reconocimiento de Patrones (Maestría FCC, BUAP).
*   **ESTADO ACTUAL:** Fase de investigación y adaptación del stack tecnológico (Hyperledger Indy/Aries) para soportar *embeddings* biométricos como credenciales verificables.

## 1. Resumen de la Arquitectura Biométrica
El sistema soluciona el riesgo de privacidad más crítico en la biometría moderna: los servidores centralizados que actúan como "honeypots" masivos de rostros y huellas. 
En este paradigma, la **plantilla biométrica (vector de características matemáticas o *embedding*)** se emite como una Credencial Verificable y reside **exclusivamente** en la billetera móvil (Wallet) del usuario.

## 2. Definición del Esquema (Schema) Biométrico
El backend emisor registrará en el ledger (DLT) un esquema diseñado para contener la firma matemática del reconocimiento de patrones:
*   **Nombre del Esquema:** `credencial-biometrica-soberana`
*   **Versión:** `1.0`
*   **Atributos:**
    *   `id_sujeto` (String - Hash anónimo del usuario)
    *   `modalidad_biometrica` (String - ej. "Rostro", "Huella_Dactilar")
    *   `algoritmo_extraccion` (String - ej. "ResNet50", "ArcFace", para garantizar compatibilidad en la comparación espacial)
    *   `template_biometrico` (String/Base64 - El vector de características n-dimensional)

## 3. Flujos de Operación (Biometría + SSI)

**Flujo A: Enrolamiento Biométrico (Emisión de la Credencial)**
1. El usuario asiste a una estación segura de enrolamiento (Autoridad Universitaria/Emisor).
2. El sistema captura el biométrico físico y lo pasa por el modelo de **Reconocimiento de Patrones**, extrayendo el *embedding* matemático (reducción de dimensionalidad).
3. Este *template* se firma criptográficamente, se empaqueta en una Credencial Verificable y se transmite vía DIDcomm a la billetera móvil del usuario.
4. **CRÍTICO:** El servidor del Emisor ELIMINA la captura física y el *template* de sus discos inmediatamente después de la emisión.

**Flujo B: Verificación Biométrica en Borde (Acceso/Torniquete)**
1. El usuario se acerca al nodo de verificación (borde).
2. El sistema solicita al usuario, vía ACA-Py, una Prueba de Conocimiento (Presentation Request) de su `template_biometrico`.
3. El usuario consiente compartir su *template* biométrico (o una ZKP booleana que prueba que es dueño de dicho *template*).
4. El Verificador de borde captura en tiempo real (*live capture*) el rostro/huella del usuario en el punto de acceso, extrae el vector en vivo y calcula la distancia espacial (ej. Distancia Euclidiana o Similitud Coseno) contra el *template* recibido en la credencial.
5. Si la distancia supera el umbral de confianza matemático, el acceso es autorizado.
6. La memoria RAM del Verificador se purga instantáneamente. Cero datos almacenados.

## 4. Justificación Científica (Investigación y Gap)
*   **Mitigación de "Honeypots" Biométricos:** Una contraseña vulnerada se cambia; un rostro vulnerado en una base centralizada está comprometido de por vida. SSI devuelve la propiedad del rasgo biológico al sujeto.
*   **Reconocimiento de Patrones Descentralizado:** Desacopla la extracción matemática de las características de su almacenamiento, demostrando que la IA y la Criptografía pueden interoperar protegiendo la privacidad.
*   **Verificación Efímera:** Al procesar la métrica de similitud en la memoria volátil del servidor de borde, el sistema previene la correlación de datos espaciales a largo plazo.

---
**Nota para agentes:** Este contexto es la base teórica para enlazar tu código de SSI (Hyperledger, Python, DIDs) con los modelos de clasificación y reconocimiento de patrones (Machine Learning). Procede a generar documentación, diagramas o papers científicos basados en esta arquitectura de soberanía biométrica.
