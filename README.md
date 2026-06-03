# Proyecto Final - Ecuaciones Diferenciales 1

## Métodos numéricos para resolver ecuaciones diferenciales ordinarias

Este proyecto corresponde a la **Alternativa 2** del proyecto final del curso de Ecuaciones Diferenciales 1. El objetivo principal es implementar y comparar dos métodos numéricos iterativos para resolver ecuaciones diferenciales ordinarias:

- Método de Euler explícito
- Método de Runge-Kutta de cuarto orden, RK4

El programa resuelve ecuaciones diferenciales de primer orden, segundo orden, sistemas lineales de 2x2 y un sistema no lineal. Para los problemas con solución analítica, se compara la aproximación numérica contra la solución exacta. Para el sistema no lineal, se analiza la convergencia usando una referencia numérica calculada con RK4 y un paso muy pequeño.

## Estructura del proyecto

```text
.
├── main.py (programa)
├── results/
│   ├── graphs/
│   │   ├── graficas generadas por el programa
│   └── tables/
│       ├── tablas CSV generadas por el programa
├── README.md
└── .gitignore
```

## Requisitos

El programa fue desarrollado en Python y se necesita instalar las siguientes librerías para que funcione

```bash
pip install numpy pandas matplotlib scipy
```

## Autor

Karen Pineda 

