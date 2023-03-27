from django import forms


class JonathanForm(forms.Form):
    design_problem = forms.CharField(
        label="",
        widget=forms.Textarea(attrs={"cols": 40, "rows": 10}),
    )
