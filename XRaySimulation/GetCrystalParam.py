import requests


def getCrystalParam(CrystalType, MillerIndex, EnergyKeV):
    """

    :param CrystalType:
    :param MillerIndex:
    :param EnergyKeV:
    :return:
    """

    ###########################################################
    #    Get response from the website
    ###########################################################
    if CrystalType in ("Silicon", "Germanium", "Diamond"):
        pass
    else:
        print("The requested crystal type is not recognized. Please check the source code.")
        return

    df1df2 = -1
    modeout = 1  # 0 - html out, 1 - quasy-text out with keywords
    detail = 0  # 0 - don't print coords, 1 = print coords

    commandline = str(r"https://x-server.gmca.aps.anl.gov/cgi/x0h_form.exe?"
                      + r"xway={}".format(2)
                      + r'&wave={}'.format(EnergyKeV)
                      + r'&line='
                      + r'&coway={}'.format(0)
                      + r'&code={}'.format(CrystalType)
                      + r'&amor='
                      + r'&chem='
                      + r'&rho='
                      + r'&i1={}'.format(MillerIndex[0])
                      + r'&i2={}'.format(MillerIndex[1])
                      + r'&i3={}'.format(MillerIndex[2])
                      + r'&df1df2={}'.format(df1df2)
                      + r'&modeout={}'.format(modeout)
                      + r'&detail={}'.format(detail))

    data = requests.get(commandline).text

    #############################################################
    #   Parse the parameters
    #############################################################
    lines = data.split("\n")
    info_holder = {}

    line_idx = 0
    total_line_num = len(lines)

    while line_idx < total_line_num:
        line = lines[line_idx]
        words = line.split()
        # print(words)
        if words:
            if words[0] == "Density":
                info_holder.update({"Density (g/cm^3)": float(words[-1])})

            elif words[0] == "Unit":
                info_holder.update({"Unit cell a1 (A)": float(words[-1])})

                # Get a2
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a2 (A)": float(words[-1])})

                # Get a3
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a3 (A)": float(words[-1])})

                # Get a4
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a4 (deg)": float(words[-1])})

                # Get a5
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a5 (deg)": float(words[-1])})

                # Get a6
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a6 (deg)": float(words[-1])})

            elif words[0] == "Poisson":
                info_holder.update({"Poisson ratio": float(words[-1])})

            elif words[0] == '<pre>':
                if words[1] == '<i>n':

                    # Get the real part of chi0
                    a = float(words[-1][4:])

                    # Get the imagniary part of chi0
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    b = float(words[-1])

                    # Add an entry
                    info_holder.update({"chi0": complex(a, b)})

                    # Skip the following two lines
                    line_idx += 2

                elif words[-1] == 'pol=Sigma':
                    # Get the real part
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    a = float(words[-1])

                    # Get the imaginary part
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    b = float(words[-1])

                    info_holder.update({"chi_s": complex(a, -b)})

                elif words[-1] == 'pol=Pi':
                    # Get the real part
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    a = float(words[-1])

                    # Get the imaginary part
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    b = float(words[-1])

                    info_holder.update({"chi_p": complex(a, -b)})
                elif words[1] == "Bragg":
                    if words[2] == "angle":
                        info_holder.update({"Bragg angle (deg)": float(words[-1])})
                else:
                    pass

            elif words[0] == 'Interplanar':
                info_holder.update({"d": float(words[-1]) * 1e-4})

            else:
                pass

        line_idx += 1

    return info_holder
