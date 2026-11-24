# App 2: Sigmoid Activation Function
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Sigmoid Activation Function")
st.write("Sigmoid formula: f(x) = 1 / (1 + e^(-x))")

# user input
x_min = st.slider("Minimum x", -20, 0, -10)
x_max = st.slider("Maximum x", 1, 20, 10)

# calculate
x = np.linspace(x_min, x_max, 200)
y = 1 / (1 + np.exp(-x))

# plot
fig, ax = plt.subplots()
ax.plot(x, y, color="green", label="Sigmoid")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("x (input)")
ax.set_ylabel("f(x) (output)")
ax.set_title("Sigmoid Graph")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# try one value
st.subheader("Try a value")
val = st.number_input("Enter x", value=2.0)
st.write("Sigmoid(", val, ") =", round(1 / (1 + np.exp(-val)), 4))

st.subheader("Explanation")
st.write("- Output is always between 0 and 1.")
st.write("- Sigmoid(0) = 0.5.")
st.write("- Often used in the output layer for binary classification.")
st.write("- Problem: vanishing gradient when x is very large or very small.")
