# 🧠 ExplicaMe — Agente adaptativo de explicaciones

Agente de IA que explica decisiones automatizadas de crédito **de forma distinta según quién las lea**, usando la misma evidencia SHAP. Demo de **explicabilidad, interpretabilidad y transparencia** para el desafío de Creación de agentes de IA adaptativos.

---

## 🎯 ¿Qué problema resuelve?

Cuando un sistema de IA rechaza una solicitud de crédito, la persona afectada no entiende **por qué**. ExplicaMe genera explicaciones adaptadas a 4 audiencias y deja **trazabilidad auditable** de cada paso.

| Audiencia | Qué recibe |
|---|---|
| 🏢 **Dirección** | Riesgos, impacto y criterios de aprobación |
| 💼 **Negocio** | Lógica general y variables clave para integrar el resultado |
| 🔬 **Técnico** | Probabilidad, contribuciones SHAP y tipo de modelo |
| 🙋 **Persona afectada** | Lenguaje claro, qué factor influyó y dónde apelar |

---

## 🏗️ Arquitectura

**Componentes:**
- **Modelo de scoring:** Regresión Logística (interpretable) y Random Forest (caja negra) para contraste.
- **SHAP:** contribuciones por variable para respaldar la decisión.
- **Agente adaptativo:** 4 prompts de sistema distintos según audiencia, con la misma evidencia como entrada.
- **Validador de fidelidad:** verifica que la explicación mencione variables reales del modelo.
- **Log auditable:** JSON con decisión, probabilidad, SHAP, versión del modelo, fuentes, hash de evidencia y timestamp.

📌 Notas
El dataset de crédito es sintético, generado con numpy.random.seed(42) para reproducibilidad.

El modelo es una demo educativa; no debe usarse en producción sin validación adicional.

Groq ofrece un tier gratuito con límites generosos para desarrollo y demos.

🙌 Créditos
Proyecto desarrollado para el desafío de Creación de agentes de IA adaptativos del curso Datos para Negocios - G11 ONE.