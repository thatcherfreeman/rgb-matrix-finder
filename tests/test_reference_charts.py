import reference_charts
import color_conversions
import numpy as np

file_contents = """

LGOROWLENGTH 12
ORIGINATOR "ColorChecker24 - November2014 edition and newer"
MANUFACTURER "X-Rite - http://www.xrite.com"

4/28/2015  # Time: 14:33
"i1Pro 2 ; Serial number 1001785"
"MeasurementCondition=M0	Filter=no"

NUMBER_OF_FIELDS 4
BEGIN_DATA_FORMAT
SAMPLE_NAME Lab_L   Lab_a   Lab_b
END_DATA_FORMAT
NUMBER_OF_SETS 24
BEGIN_DATA
A1	37,54	14,37	14,92
A2	62,73	35,83	56,5
A3	28,37	15,42	-49,8
A4	95,19	-1,03	2,93
B1	64,66	19,27	17,5
B2	39,43	10,75	-45,17
B3	54,38	-39,72	32,27
B4	81,29	-0,57	0,44
C1	49,32	-3,82	-22,54
C2	50,57	48,64	16,67
C3	42,43	51,05	28,62
C4	66,89	-0,75	-0,06
D1	43,46	-12,74	22,72
D2	30,1	22,54	-20,87
D3	81,8	2,67	80,41
D4	50,76	-0,13	0,14
E1	54,94	9,61	-24,79
E2	71,77	-24,13	58,19
E3	50,63	51,28	-14,12
E4	35,63	-0,46	-0,48
F1	70,48	-32,26	-0,37
F2	71,51	18,24	67,37
F3	49,57	-29,71	-28,32
F4	20,64	0,07	-0,46
END_DATA
"""


def test_load_reference_chart() -> None:
    lines = file_contents.split('\n')
    reference_chart, patches = reference_charts.load_reference_chart(lines)
    assert patches == (4, 6)
    assert reference_chart.colors.shape == (24, 3)
    assert np.sum(np.abs(reference_chart.colors[1] - np.array([[62.73, 35.83, 56.50]]))) < 0.0001
    assert np.sum(np.abs(reference_chart.reference_white.colors - color_conversions.STD_A.colors)) < 0.0001
