// Mały launcher: dwuklik uruchamia AutoEdit bez terminala.
// Kompilowany przez install.ps1 wbudowanym w Windows kompilatorem csc (.NET Framework 4), bez dodatkowych narzędzi.
// Uruchamia .venv\Scripts\python.exe -m autoedit.gui w ukryciu, zapisuje komunikaty do logs\autoedit.log,
// a gdy program się wysypie, pokazuje czytelny komunikat z końcówką logu zamiast milczącego zniknięcia okna.
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

static class Launcher
{
    const string Title = "AutoEdit";

    [STAThread]
    static int Main(string[] args)
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(dir, ".venv", "Scripts", "python.exe");

        if (!File.Exists(python))
        {
            MessageBox.Show(
                "Program nie jest jeszcze zainstalowany.\n\n" +
                "Uruchom plik install.bat z folderu programu (jednorazowo, potrwa kilka minut),\n" +
                "a potem uruchom AutoEdit ponownie.\n\nFolder: " + dir,
                Title, MessageBoxButtons.OK, MessageBoxIcon.Information);
            return 1;
        }

        string logDir = Path.Combine(dir, "logs");
        Directory.CreateDirectory(logDir);
        string log = Path.Combine(logDir, "autoedit.log");

        var psi = new ProcessStartInfo(python)
        {
            Arguments = "-X utf8 -m autoedit.gui" + string.Concat(args.Select(a => " \"" + a.Replace("\"", "") + "\"")),
            WorkingDirectory = dir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";

        int exit;
        using (var writer = new StreamWriter(log, false, new UTF8Encoding(false)) { AutoFlush = true })
        using (var proc = new Process { StartInfo = psi })
        {
            writer.WriteLine("=== " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " start: " + psi.FileName + " " + psi.Arguments);
            object gate = new object();
            DataReceivedEventHandler sink = (s, e) => { if (e.Data != null) lock (gate) writer.WriteLine(e.Data); };
            proc.OutputDataReceived += sink;
            proc.ErrorDataReceived += sink;
            try
            {
                proc.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Nie udało się uruchomić programu:\n\n" + ex.Message, Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            proc.WaitForExit();
            exit = proc.ExitCode;
            lock (gate) writer.WriteLine("=== koniec, kod wyjścia " + exit);
        }

        if (exit != 0)
        {
            string tail = "";
            try { tail = string.Join("\n", File.ReadAllLines(log, Encoding.UTF8).Reverse().Take(14).Reverse()); } catch { }
            MessageBox.Show(
                "AutoEdit zakończył pracę z błędem (kod " + exit + ").\n\n" + tail +
                "\n\nPełny zapis: " + log,
                Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        return exit;
    }
}
