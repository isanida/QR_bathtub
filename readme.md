# Qualitative Reasoning - KR Project


**Quantities**
- Inflow 
- Volume 
- Outflow 
– Height
– Pressure

**Quantity spaces**
- Inflow: [0, +]
- Outflow: [0, +, max]
- Volume: [0, +, max]
– Height: [0, +, max]
– Pressure: [0, +, max]

**Dependencies**
– I+(Inflow, Volume)
– I-(Outflow, Volume)
– P+(Volume, Outflow)
– VC(Volume(max), Outflow(max))
– VC(Volume(0), Outflow(0))
– P+(Volume, Pressure)
– P+(Volume, Height)
– VC(Volume(max), Pressure(max))
– VC(Volume(0), Pressure(0))
– VC(Volume(max), Height(max))
– VC(Volume(0), Height(0))
