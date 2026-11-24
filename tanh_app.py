# App 3: Tanh Activation Function
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Hyperbolic Tangent (Tanh) Activation Function")
st.write("Tanh formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))")

# user input
x_min = st.slider("Minimum x", -20, 0, -10)
x_max = st.slider("Maximum x", 1, 20, 10)

# calculate
x = np.linspace(x_min, x_max, 200)
y = np.tanh(x)

# plot
fig, ax = plt.subplots()
ax.plot(x, y, color="red", label="Tanh")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("x (input)")
ax.set_ylabel("f(x) (output)")
ax.set_title("Tanh Graph")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# try one value
st.subheader("Try a value")
val = st.number_input("Enter x", value=2.0)
st.write("Tanh(", val, ") =", round(float(np.tanh(val)), 4))

st.subheader("Explanation")
st.write("- Output is always between -1 and 1.")
st.write("- Tanh(0) = 0 (zero-centered, unlike Sigmoid).")
st.write("- Usually works better than Sigmoid in hidden layers.")
st.write("- Problem: still has vanishing gradient for large |x|.")
