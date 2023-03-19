from django import forms


class JonathanForm(forms.Form):
    design_problem = forms.CharField(
        label="",
        widget=forms.Textarea(attrs={"cols": 40, "rows": 10}),
    )


class SubmitPatternForm(forms.Form):
    content = forms.CharField(label="", widget=forms.Textarea(attrs={}))


class CollectPatternForm(forms.Form):
    urlContent = forms.CharField(label="")
    sectionContent1 = forms.CharField(label="")
    sectionContent2 = forms.CharField(label="")
