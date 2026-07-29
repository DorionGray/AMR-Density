from django import forms

_num = lambda step='any': forms.NumberInput(attrs={'class': 'form-input', 'step': step})


class SIRForm(forms.Form):
    # ── Disease dynamics ────────────────────────────────────────────────
    k = forms.FloatField(label='Contact rate (k)', initial=12.413, min_value=1e-8, widget=_num())
    v = forms.FloatField(label='Transmission probability (v)', initial=0.0014, min_value=1e-8, max_value=1.0, widget=_num())
    mu = forms.FloatField(label='Mortality rate (μ)', initial=0.035, min_value=0.0, max_value=1.0, widget=_num())
    gamma = forms.FloatField(label='Natural recovery rate (γ)', initial=0.167, min_value=0.0, widget=_num())
    gamma_a = forms.FloatField(label='Antibiotic recovery rate (γₐ)', initial=0.379, min_value=0.0, widget=_num())
    epson = forms.FloatField(label='De novo resistance probability (ε)', initial=0.0667, min_value=0.0, max_value=1.0, widget=_num())
    alpha = forms.FloatField(label='Fitness cost of resistance (α)', initial=0.5, min_value=0.0, max_value=1.0, widget=_num())

    # ── Treatment policy ────────────────────────────────────────────────
    p = forms.FloatField(label='Mortality threshold for treatment (p)', initial=0.02, min_value=0.0, max_value=1.0, widget=_num())
    day_max = forms.IntegerField(label='Last day treatment can start', initial=35, min_value=1, max_value=42, widget=_num(step=1))
    IS0_relative = forms.FloatField(label='Initial sensitive-infected fraction (IS₀)', initial=0.05452, min_value=0.0, max_value=1.0, widget=_num())
    IR0_relative = forms.FloatField(label='Initial resistant-infected fraction (IR₀)', initial=0.00348, min_value=0.0, max_value=1.0, widget=_num())

    # ── Density range ───────────────────────────────────────────────────
    density_min = forms.FloatField(label='Min stocking density (animals/m²)', initial=5.0, min_value=0.01, widget=_num())
    density_max = forms.FloatField(label='Max stocking density (animals/m²)', initial=25.0, min_value=0.01, widget=_num())
    current_density = forms.FloatField(label='Current stocking density (animals/m²)', initial=21.0, min_value=0.01, widget=_num())

    # ── Farm & economics ────────────────────────────────────────────────
    A = forms.FloatField(label='Farm area (m²)', initial=1996.4, min_value=1.0, widget=_num())
    W = forms.FloatField(label='Healthy bird weight (kg)', initial=2.404, min_value=0.01, widget=_num())
    P = forms.FloatField(label='Price per kg (€)', initial=0.83, min_value=0.001, widget=_num())
    Wloss = forms.FloatField(label='Weight loss – (fraction)', initial=0.2080, min_value=0.0, max_value=1.0, widget=_num())

    average_expenditure = forms.FloatField(label='Maintenance cost (€/animal/day)', initial=0.0071, min_value=0.0, widget=_num())
    antibiotic_cost = forms.FloatField(label='Antibiotic cost (€/animal/day)', initial=0.0019, min_value=0.0, widget=_num())

    def clean(self):
        cleaned = super().clean()
        d_min = cleaned.get('density_min')
        d_max = cleaned.get('density_max')
        if d_min is not None and d_max is not None and d_min >= d_max:
            raise forms.ValidationError('Min density must be less than max density.')
        cur = cleaned.get('current_density')
        if cur is not None and d_min is not None and d_max is not None and not (d_min <= cur <= d_max):
            raise forms.ValidationError('Current density must be between min and max density.')
        is0 = cleaned.get('IS0_relative', 0)
        ir0 = cleaned.get('IR0_relative', 0)
        if is0 is not None and ir0 is not None and is0 + ir0 >= 1.0:
            raise forms.ValidationError('IS₀ + IR₀ must be less than 1.')
        return cleaned
