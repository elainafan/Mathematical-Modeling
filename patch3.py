import re

with open("Q4/q4_static_flow_simulation.py", "r", encoding="utf-8") as f:
    text = f.read()

new_main = """    cities = ["Chengdu", "Dalian", "Dongguan", "Harbin", "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou"]
    fractions = np.linspace(0.02, 0.30, 15)
    
    out_dir = os.path.join(os.path.dirname(__file__), "Results")
    os.makedirs(out_dir, exist_ok=True)
    
    import matplotlib.pyplot as plt
    
    for city in cities:
        print(f"\\n{'='*50}\\n====== Processing {city} ======\\n{'='*50}")
        df_flow = simulate_static_deletion_flow(city, fractions, metric_type="initial_flow")
        
        df_flow.to_csv(os.path.join(out_dir, f"StaticFlow_{city}_Flow.csv"), index=False)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1) Accessibility
        axes[0].plot(df_flow['Fraction'] * 100, df_flow['Accessibility'], marker='o', label='Static (Initial Flow)')
        axes[0].set_title(f'Accessibility $A$ Decay ({city})')
        axes[0].set_xlabel('Nodes Deleted (%)')
        axes[0].set_ylabel('Accessibility A')
        axes[0].legend()
        axes[0].grid(True, linestyle="--")
        
        # 2) Rho_Max
        axes[1].plot(df_flow['Fraction'] * 100, df_flow['Rho_Max'], marker='o', label='Max $\\\\rho$ (Initial Flow)')
        axes[1].set_title(f'Max Congestion $\\\\rho$ ({city})')
        axes[1].set_xlabel('Nodes Deleted (%)')
        axes[1].set_ylabel('Max Congestion $\\\\rho$')
        axes[1].legend()
        axes[1].grid(True, linestyle="--")
        
        # 3) Gini
        axes[2].plot(df_flow['Fraction'] * 100, df_flow['Gini'], marker='o', label='Gini (Initial Flow)')
        axes[2].set_title(f'Flow Gini Coefficient ({city})')
        axes[2].set_xlabel('Nodes Deleted (%)')
        axes[2].set_ylabel('Gini Index')
        axes[2].legend()
        axes[2].grid(True, linestyle="--")
        
        plt.tight_layout()
        pdf_path = os.path.join(out_dir, f"Q4_Static_Flow_Deletion_{city}.pdf")
        plt.savefig(pdf_path)
        plt.close(fig)
        print(f"[{city}] Completed. Saved to {pdf_path}")
"""

text = re.sub(r'if __name__ == "__main__":\n(.*)', f'if __name__ == "__main__":\n{new_main}', text, flags=re.DOTALL)

with open("Q4/q4_static_flow_simulation.py", "w", encoding="utf-8") as f:
    f.write(text)
