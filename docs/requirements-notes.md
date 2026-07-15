# Requirements Notes

Source artifacts currently in workspace:

- `D1.1_AURORA_System_Specification_v4.docx`
- `Hirvensalmi_tuotanto2025.xlsx`

Initial interpretation:

- Forecast target: solar/PV production at pilot sites.
- Time grain: 15-minute intervals in the Hirvensalmi workbook.
- Forecast model path: baseline models first, then Temporal Fusion Transformer.
- Optimization path: MPC for battery energy storage charge-discharge scheduling.
- Metrics: MAE, RMSE, MBE, plus pilot KPIs from the system specification.

Open questions:

- Canonical site metadata schema.
- Source and access pattern for satellite/NWP inputs.
- Battery capacity and power constraints for the Finnish pilot.
- Whether linked source workbooks referenced by `Hirvensalmi_tuotanto2025.xlsx` are available.
