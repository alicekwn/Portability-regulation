import numpy as np
import matplotlib.pyplot as plt
from portability_env.settings import PLOT_PATH


class Bank:
    def __init__(self, a1, a2, a3, a4, b1, b2, b3, rI):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.rI = rI

    def s_star(self, T: float) -> float:
        numerator = -self.a2 * self.a3 * self.b1 + np.sqrt(
            self.a2
            * (self.a3 * self.b1 - self.a1 * (self.b3 + self.a4 * T))
            * (
                self.a2 * self.a3 * self.b1
                + self.a1 * (self.rI - self.b2)
                - self.a1 * self.a2 * (self.b3 + self.a4 * T)
            )
        )
        denominator = self.a1 * self.a2 * self.a3
        output = numerator / denominator
        output = np.where((output >= 0) & (output <= 1), output, np.nan)
        return output

    def D(self, T: float) -> float:
        output = self.a3 * self.s_star(T) + self.a4 * T + self.b3
        output = np.where((output >= 0), output, np.nan)
        return output

    def rD(self, T: float) -> float:
        output = self.a2 * self.D(T) + self.b2
        output = np.where((output >= 0) & (output < self.rI), output, np.nan)
        return output

    def c(self, T: float) -> float:
        output = self.a1 * self.s_star(T) + self.b1
        output = np.where((output > 0), output, np.nan)
        return output

    def rB(self, T: float) -> float:
        output = (self.rI - self.rD(T)) * self.D(T) / self.c(T)
        output = np.where((output >= 0), output, np.nan)
        return output

    def dr_dT(self, T: float) -> float:
        numerator = (self.rI - self.rD(T) - self.D(T) * self.a2) * self.a4
        denominator = self.c(T)
        output = numerator / denominator
        return output


if __name__ == "__main__":
    T = np.linspace(0, 1, 100)

    # ----- Bank -----
    bank_g = Bank(
        a1=0.2,
        a2=0.1,
        a3=5.87,
        a4=1.46,
        b1=0.01,
        b2=-0.05,
        b3=-4.47,
        rI=0.1,
    )
    bank_s = Bank(
        a1=0.2,
        a2=0.1,
        a3=3.3,
        a4=-0.31,
        b1=0.01,
        b2=-0.05,
        b3=-0.9,
        rI=0.1,
    )
    # Plot V* vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g.s_star(T), label=r"$S>\bar{S}$", color="blue")
    ax.plot(T, bank_s.s_star(T), label=r"$S<\bar{S}$", color="orange")
    ax.axhline(0.6, label=r"$\bar{S}$", color="black", linestyle="--")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$S^{*}(T)$")
    ax.set_title(r"$S^{*}(T)$ vs $T$")
    plt.savefig(PLOT_PATH / "bank_sstar.png")
    plt.show()

    # Plot C vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g.c(T), label=r"$S>\bar{S},$", color="blue")
    ax.plot(T, bank_s.c(T), label=r"$S<\bar{S}$", color="orange")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$C(T)$")
    ax.set_title(r"$C(T)$ vs $T$")
    plt.savefig(PLOT_PATH / "bank_c.png")
    plt.show()

    # Plot dR/dT vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g.dr_dT(T), label=r"$S>\bar{S}$", color="blue")
    ax.plot(T, bank_s.dr_dT(T), label=r"$S<\bar{S}$", color="orange")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$dR^{B}/dT$")
    ax.set_title(r"$dR^{B}/dT$ vs $T$")
    plt.savefig(PLOT_PATH / "bank_drdT.png")
    plt.show()

    # ----- Bank when C -> 0 -----
    bank_g_0 = Bank(
        a1=0.001,
        a2=0.1,
        a3=5.87,
        a4=1.46,
        b1=0,
        b2=-0.05,
        b3=-4.47,
        rI=0.1,
    )
    bank_s_0 = Bank(
        a1=0.001,
        a2=0.1,
        a3=3.3,
        a4=-0.31,
        b1=0,
        b2=-0.05,
        b3=-0.9,
        rI=0.1,
    )
    # Plot V* vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g_0.s_star(T), label=r"$S>\bar{S}, C \to 0$", color="blue")
    ax.plot(T, bank_s_0.s_star(T), label=r"$S<\bar{S}, C \to 0$", color="orange")
    ax.axhline(0.6, label=r"$\bar{S}$", color="black", linestyle="--")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$S^{*}(T)$")
    ax.set_title(r"$S^{*}(T)$ vs $T$ when $C \to 0$")
    plt.savefig(PLOT_PATH / "bank_sstar_c_0.png")
    plt.show()

    # Plot C vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g_0.c(T), label=r"$S>\bar{S}$", color="blue")
    ax.plot(T, bank_s_0.c(T), label=r"$S<\bar{S}$", color="orange")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$C(T)$")
    ax.set_title(r"$C(T)$ vs $T$")
    plt.savefig(PLOT_PATH / "bank_c_0.png")
    plt.show()

    # Plot dR/dT vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g_0.dr_dT(T), label=r"$S>\bar{S}, C \to 0$", color="blue")
    ax.plot(T, bank_s_0.dr_dT(T), label=r"$S<\bar{S}, C \to 0$", color="orange")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$dR^{B}/dT$")
    ax.set_title(r"$dR^{B}/dT$ vs $T$ when $C \to 0$")
    plt.savefig(PLOT_PATH / "bank_drdT_c_0.png")
    plt.show()

    # ----- Bank when C -> inf-----
    bank_g_inf = Bank(
        a1=10,
        a2=0.1,
        a3=5.87,
        a4=1.46,
        b1=10,
        b2=-0.05,
        b3=-4.47,
        rI=0.1,
    )
    bank_s_inf = Bank(
        a1=10,
        a2=0.1,
        a3=3.3,
        a4=-0.31,
        b1=10,
        b2=-0.05,
        b3=-0.9,
        rI=0.1,
    )
    # Plot V* vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g_inf.s_star(T), label=r"$S>\bar{S}, C \to \infty$", color="blue")
    ax.plot(T, bank_s_inf.s_star(T), label=r"$S<\bar{S}, C \to \infty$", color="orange")
    ax.axhline(0.6, label=r"$\bar{S}$", color="black", linestyle="--")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$S^{*}(T)$")
    ax.set_title(r"$S^{*}(T)$ vs $T$ when $C \to \infty$")
    plt.savefig(PLOT_PATH / "bank_sstar_c_inf.png")
    plt.show()

    # Plot C vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g_inf.c(T), label=r"$S>\bar{S}$", color="blue")
    ax.plot(T, bank_s_inf.c(T), label=r"$S<\bar{S}$", color="orange")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$C(T)$")
    ax.set_title(r"$C(T)$ vs $T$")
    plt.savefig(PLOT_PATH / "bank_c_inf.png")
    plt.show()

    # Plot dR/dT vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, bank_g_inf.dr_dT(T), label=r"$S>\bar{S}, C \to \infty$", color="blue")
    ax.plot(T, bank_s_inf.dr_dT(T), label=r"$S<\bar{S}, C \to \infty$", color="orange")
    ax.legend()
    ax.set_xlabel("Portability (T)")
    ax.set_ylabel(r"$dR^{B}/dT$")
    ax.set_title(r"$dR^{B}/dT$ vs $T$ when $C \to \infty$")
    plt.savefig(PLOT_PATH / "bank_drdT_c_inf.png")
    plt.show()
