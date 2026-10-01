# 📄 Reporte de Proyecto: Artículo Científico

Este directorio contiene el código fuente en **LaTeX** utilizado para redactar el artículo científico que documenta todo el pipeline de Machine Learning (preprocesamiento, normalización, correlación y partición) de la Tarea 01 de Reconocimiento de Patrones.

## 📂 Contenido y Archivos `.tex`

La plantilla de LaTeX ha sido modularizada para mantener el contenido limpio y organizado al momento de redactar. Aquí tienes la descripción de cada archivo:

*   **`main.tex`**: Es el archivo maestro o "enrutador". Aquí se define el Título del artículo, el Nombre del Autor, el *Abstract* (Resumen) y las Palabras Clave. Su función principal es mandar a llamar a los demás capítulos en orden y generar el esqueleto general. Para compilar el proyecto, este es el archivo principal que se ejecuta.
*   **`01-Intro.tex`**: Contiene la **Introducción**. Plantea el problema general del ruido en los datos crudos y la justificación de por qué el preprocesamiento y la dimensionalidad son críticos antes de usar modelos matemáticos.
*   **`02-Background.tex`**: Es el **Marco Teórico**. Explica las teorías, matemáticas y el propósito algorítmico detrás de las herramientas utilizadas (Min-Max, Z-Score, Robust, Pearson, PCA, CCA).
*   **`03-Method.tex`**: Es la **Metodología**. Describe literalmente lo que hace nuestro orquestador `main.py` paso a paso (desde leer los `.txt` hasta la estratificación y partición 70/15/15).
*   **`04-Experimental.tex`**: Es la sección de **Resultados y Experimentación**. Demuestra con los números reales extraídos de la terminal qué tan efectivo fue el script (ej. reducción explícita de "columnas clonadas" y los porcentajes de varianza retenida por PCA).
*   **`05-Conclusions.tex`**: Las **Conclusiones** finales del artículo donde fundamentamos por qué nuestros datasets finales quedaron óptimos para el proceso de entrenamiento.
*   **`biblio.bib`**: Es el archivo de base de datos de **BibTeX** que guarda toda nuestra Bibliografía. Contiene los cuatro pilares fundamentales de la literatura de Reconocimiento de Patrones (libros de Duda, Bishop, Fieguth y el paper de Jain).

## 🛠️ Compilación

Para compilar manualmente estos archivos y generar el PDF final actualizado, abre una terminal en esta carpeta y ejecuta los siguientes comandos (asegúrate de tener una distribución LaTeX instalada):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

El resultado final se actualizará en el archivo `main.pdf`.
