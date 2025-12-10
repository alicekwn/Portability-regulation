import numpy as np
import matplotlib.pyplot as plt


class Insurer:
    def __init__(self, a1, a2, a3, a4, a5, b1, b2, rI, rG, k):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.b1 = b1
        self.b2 = b2
        self.rI = rI
        self.rG = rG
        self.k = k

    def v_star(self, T: float) -> float:
        numerator = self.a3 * (self.b1 + self.b1 * self.k * self.rI) + np.sqrt(
            self.a3
            * (1 + self.k * self.rI)
            * (
                -self.a1 * self.a2 * self.b1 * (1 + self.k * self.rI)
                + self.a3 * self.b1**2 * (1 + self.k * self.rI)
                + self.a1**2
                * (
                    self.b2
                    + self.b2 * self.k * self.rI
                    + (self.a4 - self.a5 + self.a4 * self.k * self.rI) * T
                )
            )
        )
        denominator = -self.a1 * self.a3 * (1 + self.k * self.rI)
        output = numerator / denominator
        output = np.where((output >= 0) & (output <= 1), output, np.nan)
        return output

    def c(self, T: float) -> float:
        return self.a1 * T + self.b1

    def l(self, T: float) -> float:
        return self.a5 * T

    def p(self, T: float) -> float:
        output = (
            self.a2 * self.v_star(T)
            + self.a3 * self.v_star(T) ** 2
            + self.a4 * T
            + self.b2
        )
        output = np.where(
            (output >= 0) & (self.a2 + 2 * self.a3 * self.v_star(T) >= 0),
            output,
            np.nan,
        )
        return output

    def r(self, T: float) -> float:
        return self.rG + ((1 + self.k * self.rI) + self.p(T) - self.l(T)) / self.c(
            self.v_star(T)
        )

    def dr_dT(self, T: float) -> float:
        numerator = (1 + self.k * self.rI) * self.a4 - self.a5
        denominator = self.c(self.v_star(T))
        output = numerator / denominator
        return output


if __name__ == "__main__":
    T = np.linspace(0, 1, 100)

    # ----- Insurer when C -> 0 -----
    insurer_g_0 = Insurer(
        a1=0.001,
        a2=3.2,
        a3=-1.8,
        a4=3.4,
        a5=3.395,
        b1=0.0001,
        b2=-1.3,
        rI=0.13,
        rG=0.1,
        k=3,
    )
    insurer_s_0 = Insurer(
        a1=0.001,
        a2=6.2,
        a3=-1.9,
        a4=-1.6,
        a5=-1.595,
        b1=0.0001,
        b2=0.6,
        rI=0.13,
        rG=0.1,
        k=3,
    )
    # Plot V* vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, insurer_g_0.v_star(T), label=r"$V>\bar{V}, C \to 0$", color="blue")
    ax.plot(T, insurer_s_0.v_star(T), label=r"$V<\bar{V}, C \to 0$", color="orange")
    ax.axhline(0.45, label=r"$\bar{V}$", color="black", linestyle="--")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$V^{*}(T)$")
    plt.show()

    # Plot dR/dT vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, insurer_g_0.dr_dT(T), label=r"$V>\bar{V}, C \to 0$", color="blue")
    ax.plot(T, insurer_s_0.dr_dT(T), label=r"$V<\bar{V}, C \to 0$", color="orange")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"d$R^{IC}$/dT")
    plt.show()

    # ----- Insurer when C -> inf -----
    insurer_g_inf = Insurer(
        a1=1000,
        a2=4.5,
        a3=-2.53,
        a4=7,
        a5=6.995,
        b1=1000,
        b2=-0.96,
        rI=0.13,
        rG=0.1,
        k=3,
    )
    insurer_s_inf = Insurer(
        a1=1000,
        a2=1.55,
        a3=-0.65,
        a4=-0.85,
        a5=-0.845,
        b1=1000,
        b2=1.35,
        rI=0.13,
        rG=0.1,
        k=3,
    )
    # Plot V* vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, insurer_g_inf.v_star(T), label=r"$V>\bar{V}, C \to inf$", color="blue")
    ax.plot(T, insurer_s_inf.v_star(T), label=r"$V<\bar{V}, C \to inf$", color="orange")
    ax.axhline(0.45, label=r"$\bar{V}$", color="black", linestyle="--")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$V^{*}(T)$")
    plt.show()

    # Plot dR/dT vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, insurer_g_inf.dr_dT(T), label=r"$V>\bar{V}, C \to inf$", color="blue")
    ax.plot(T, insurer_s_inf.dr_dT(T), label=r"$V<\bar{V}, C \to inf$", color="orange")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"d$R^{IC}$/dT")
    plt.show()
