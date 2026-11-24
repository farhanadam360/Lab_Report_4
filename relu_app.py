# App 1: ReLU Activation Function
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("ReLU Activation Function")
st.write("ReLU (Rectified Linear Unit) formula: f(x) = max(0, x)")

# user input
x_min = st.slider("Minimum x", -20, 0, -10)
x_max = st.slider("Maximum x", 1, 20, 10)

# calculate
x = np.linspace(x_min, x_max, 200)
y = np.maximum(0, x)

# plot
fig, ax = plt.subplots()
ax.plot(x, y, color="blue", label="ReLU")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("x (input)")
ax.set_ylabel("f(x) (output)")
ax.set_title("ReLU Graph")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# try one value
st.subheader("Try a value")
val = st.number_input("Enter x", value=2.0)
st.write("ReLU(", val, ") =", max(0, val))

st.subheader("Explanation")
st.write("- If x is negative, the output is 0.")
st.write("- If x is positive, the output is the same as x.")
st.write("- Range of output: 0 to infinity.")
st.write("- Very fast to compute and commonly used in hidden layers.")
st.write("- Problem: 'dying ReLU' (neurons that always output 0).")
