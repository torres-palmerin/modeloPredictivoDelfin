"""
Módulo del motor del autómata finito para trayectorias académicas (Fase 2).
Implementa un autómata determinista que modela los estados institucionales
y transiciones de cada estudiante a lo largo de su historial académico.
"""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Estado inicial de todo estudiante al ingresar al sistema
ESTADO_INICIAL = "Aspirante inscrito"


class AcademicAutomaton:
    """
    Motor del Autómata Finito para la construcción de Trayectorias Académicas.
    Sigue las reglas institucionales de estados y transiciones.

    States:
        - Aspirante inscrito
        - Primera vez en una carrera
        - Continuo regular
        - PAP (Prueba Académica Parcial)
        - PAT (Prueba Académica Total)
        - Recuperación académica
        - Exclusión
        - Transferencia interna
        - Por fuera de la universidad
    """

    # Estados que representan situación académica activa
    ESTADOS_ACTIVOS = frozenset({
        "Primera vez en una carrera",
        "Continuo regular",
        "PAP",
        "PAT",
        "Recuperación académica",
    })

    # Mapa de transiciones: (estado_actual, condición) -> (nuevo_estado, símbolo)
    TRANSICIONES = {
        "Continuo regular": {
            "regular": ("Continuo regular", "a"),
            "pap": ("PAP", "b"),
        },
        "Primera vez en una carrera": {
            "regular": ("Continuo regular", "a"),
            "pap": ("PAP", "b"),
        },
        "PAP": {
            "recupera": ("Continuo regular", "a"),
            "pap": ("PAT", "b"),
        },
        "PAT": {
            "recupera": ("Continuo regular", "a"),
            "recuperacion": ("Recuperación académica", "e"),
            "exclusion": ("Exclusión", "d"),
        },
        "Recuperación académica": {
            "recupera": ("Continuo regular", "a"),
            "exclusion": ("Exclusión", "d"),
        },
    }

    def __init__(
        self,
        ppp_threshold: float = 3.2,
        ppa_threshold: float = 3.2,
    ):
        """
        Inicializa el autómata con umbrales configurables.

        Args:
            ppp_threshold: Umbral mínimo del Promedio del Periodo para estar regular.
            ppa_threshold: Umbral mínimo del Promedio Acumulado para estar regular.
        """
        if ppp_threshold <= 0 or ppa_threshold <= 0:
            raise ValueError("Los umbrales deben ser valores positivos.")
        self.ppp_threshold = ppp_threshold
        self.ppa_threshold = ppa_threshold
        logger.info(
            "Autómata inicializado — Umbral PPP: %.2f, Umbral PPA: %.2f",
            self.ppp_threshold,
            self.ppa_threshold,
        )

    def _clasificar_periodo(self, ppp: float, ppa: float) -> str:
        """
        Clasifica el desempeño de un periodo en una categoría de transición.

        Returns:
            Cadena clave que indexa el diccionario TRANSICIONES.
        """
        regular = ppp >= self.ppp_threshold and ppa >= self.ppa_threshold
        if regular:
            return "regular"
        if ppp >= self.ppp_threshold and ppa < self.ppa_threshold:
            return "recuperacion"
        if ppp >= self.ppp_threshold:
            return "recupera"
        # ppp por debajo del umbral
        return "pap" if ppa >= self.ppa_threshold else "exclusion"

    def _determinar_siguiente_estado(
        self,
        estado_actual: str,
        clasificacion: str,
        es_primer_periodo: bool,
    ) -> Tuple[str, str]:
        """
        Determina el siguiente estado y símbolo de transición del autómata.

        Args:
            estado_actual: Estado académico actual del estudiante.
            clasificacion: Clasificación del periodo (regular, pap, etc.).
            es_primer_periodo: True si es el primer periodo registrado.

        Returns:
            Tupla (nuevo_estado, símbolo_transición).
        """
        # Caso especial: primer periodo
        if es_primer_periodo and estado_actual == ESTADO_INICIAL:
            return "Primera vez en una carrera", "n"

        # Buscar transición en el mapa
        transiciones_estado = self.TRANSICIONES.get(estado_actual, {})
        resultado = transiciones_estado.get(clasificacion)

        if resultado is not None:
            return resultado

        # Mantener estado actual si no hay regla específica aplicable
        return estado_actual, "a"

    def build_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa el DataFrame y genera las trayectorias académicas de cada estudiante.

        Clasifica vectorizadamente cada periodo y luego procesa por grupo de
        estudiante para calcular la secuencia de estados y transiciones.
        Los resultados se recolectan en listas posicionales para garantizar
        la alineación correcta con el DataFrame original.

        Args:
            df: DataFrame limpio con columnas [ID, PERIODO, PPP, PPA, ...].

        Returns:
            DataFrame original más las columnas AUTOMATA_ESTADO_MATH y TRANSICION_MATH.
        """
        logger.info("Fase 2 — Construyendo trayectorias del autómata...")

        df = df.sort_values(by=['ID', 'PERIODO']).copy()

        # Precalcular columnas auxiliares vectorizadas
        df['_periodo_min'] = df.groupby('ID')['PERIODO'].transform('min')
        df['_es_primer_periodo'] = df['PERIODO'] == df['_periodo_min']

        # Clasificar cada periodo vectorizadamente
        df['_clasificacion'] = np.select(
            [
                (df['PPP'] >= self.ppp_threshold) & (df['PPA'] >= self.ppa_threshold),
                (df['PPP'] >= self.ppp_threshold) & (df['PPA'] < self.ppa_threshold),
                df['PPP'] >= self.ppp_threshold,
            ],
            ['regular', 'recuperacion', 'recupera'],
            default='pap',
        )
        # Corregir exclusión: ppp < umbral Y ppa < umbral
        mask_exclusion = (df['PPP'] < self.ppp_threshold) & (df['PPA'] < self.ppa_threshold)
        df.loc[mask_exclusion, '_clasificacion'] = 'exclusion'

        # Procesamiento por grupo: iterar sobre grupos (no filas) para mantener
        # la secuencia de estados y recolectar resultados en listas posicionales.
        # Esto evita el bug de desalineación de índices que produce groupby().apply().
        estados_all: List[str] = []
        transiciones_all: List[str] = []

        for _, grupo in df.groupby('ID', sort=False):
            estados_grupo, transiciones_grupo = self._procesar_grupo_estudiante(grupo)
            estados_all.extend(estados_grupo)
            transiciones_all.extend(transiciones_grupo)

        df['AUTOMATA_ESTADO_MATH'] = estados_all
        df['TRANSICION_MATH'] = transiciones_all

        # Limpiar columnas auxiliares
        df.drop(columns=['_periodo_min', '_es_primer_periodo', '_clasificacion'], inplace=True)

        total_estudiantes = df['ID'].nunique()
        logger.info(
            "Fase 2 completada — %d registros, %d estudiantes procesados",
            len(df),
            total_estudiantes,
        )
        return df

    def _procesar_grupo_estudiante(
        self, grupo: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """
        Procesa la secuencia de periodos de un solo estudiante.
        Aplicado vía groupby().apply() para evitar iterrows().
        """
        estados: List[str] = []
        transiciones: List[str] = []
        estado_actual = ESTADO_INICIAL

        for es_primer, clasif in zip(
            grupo['_es_primer_periodo'],
            grupo['_clasificacion'],
        ):
            nuevo_estado, transicion = self._determinar_siguiente_estado(
                estado_actual=estado_actual,
                clasificacion=clasif,
                es_primer_periodo=es_primer,
            )
            estados.append(nuevo_estado)
            transiciones.append(transicion)
            estado_actual = nuevo_estado

        return estados, transiciones
