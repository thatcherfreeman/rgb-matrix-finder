import src.reference_charts as reference_charts
import src.color_conversions as color_conversions
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

file_contents2 = """
Date: 3/27/2000  Time: 19:04
LGOROWLENGTH 12
ORIGINATOR "ColorChecker24 - Before November2014 edition"
MANUFACTURER "X-Rite - http://www.xrite.com"
NUMBER_OF_FIELDS 4
BEGIN_DATA_FORMAT
SAMPLE_NAME Lab_L   Lab_a   Lab_b
END_DATA_FORMAT
NUMBER_OF_SETS 24
BEGIN_DATA
A1	37.986	13.555	14.059
A2	62.661	36.067	57.096
A3	28.778	14.179	-50.297
A4	96.539	-0.425	1.186
B1	65.711	18.13	17.81
B2	40.02	10.41	-45.964
B3	55.261	-38.342	31.37
B4	81.257	-0.638	-0.335
C1	49.927	-4.88	-21.905
C2	51.124	48.239	16.248
C3	42.101	53.378	28.19
C4	66.766	-0.734	-0.504
D1	43.139	-13.095	21.905
D2	30.325	22.976	-21.587
D3	81.733	4.039	79.819
D4	50.867	-0.153	-0.27
E1	55.112	8.844	-25.399
E2	72.532	-23.709	57.255
E3	51.935	49.986	-14.574
E4	35.656	-0.421	-1.231
F1	70.719	-33.397	-0.199
F2	71.941	19.363	67.857
F3	51.038	-28.631	-28.638
F4	20.461	-0.079	-0.973
END_DATA
"""


def test_load_reference_chart() -> None:
    lines = file_contents.split("\n")
    reference_chart, patches = reference_charts.load_reference_chart(lines)
    assert patches == (4, 6)
    assert reference_chart.colors.shape == (24, 3)
    assert (
        np.sum(np.abs(reference_chart.colors[1] - np.array([[62.73, 35.83, 56.50]])))
        < 0.0001
    )
    assert (
        np.sum(
            np.abs(
                reference_chart.reference_white.colors - color_conversions.STD_A.colors
            )
        )
        < 0.0001
    )


def test_load_reference_chart2() -> None:
    lines = file_contents2.split("\n")
    reference_chart, patches = reference_charts.load_reference_chart(lines)
    assert patches == (4, 6)
    assert reference_chart.colors.shape == (24, 3)
    assert (
        np.sum(np.abs(reference_chart.colors[1] - np.array([[62.661, 36.067, 57.096]])))
        < 0.0001
    )
    assert (
        np.sum(
            np.abs(
                reference_chart.reference_white.colors - color_conversions.STD_A.colors
            )
        )
        < 0.0001
    )
