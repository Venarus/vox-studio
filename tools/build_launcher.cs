using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

namespace VoxStudioLauncher
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            try
            {
                // Active workspace on Desktop: VOX Studio — Audio Engine
                string studioDir = @"C:\Users\senem\Desktop\VOX Studio — Audio Engine";
                string scriptPath = Path.Combine(studioDir, "main.py");

                if (!File.Exists(scriptPath))
                {
                    MessageBox.Show(
                        "VOX Studio Audio Engine ana dosyası bulunamadı:\n" + scriptPath,
                        "VOX Studio — Audio Engine",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return;
                }

                // Python yürütücüsünü bul
                string pythonExe = @"C:\Users\senem\AppData\Local\Programs\Python\Python313\pythonw.exe";
                if (!File.Exists(pythonExe))
                {
                    pythonExe = @"C:\Users\senem\AppData\Local\Programs\Python\Python313\python.exe";
                }
                if (!File.Exists(pythonExe))
                {
                    pythonExe = "pythonw.exe";
                }

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonExe;
                psi.Arguments = "\"" + scriptPath + "\"";
                if (args.Length > 0)
                {
                    psi.Arguments += " " + string.Join(" ", args);
                }
                psi.WorkingDirectory = studioDir;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                psi.RedirectStandardError = true;
                psi.StandardErrorEncoding = Encoding.UTF8;

                Process proc = Process.Start(psi);
                if (proc == null)
                {
                    MessageBox.Show("VOX Studio işlemi başlatılamadı.", "VOX Studio", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                // Erken çökme denetimi (2.0 saniye bekle)
                if (proc.WaitForExit(2000))
                {
                    if (proc.ExitCode != 0)
                    {
                        string stderr = proc.StandardError.ReadToEnd();
                        string msg = string.Format(
                            "VOX Studio başlatılırken bir hata oluştu (Çıkış Kodu: {0}):\n\n{1}",
                            proc.ExitCode,
                            string.IsNullOrEmpty(stderr) ? "Ayrıntı yok. Lütfen log dosyasını kontrol edin." : stderr
                        );
                        MessageBox.Show(msg, "VOX Studio — Başlatma Hatası", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "VOX Studio başlatma hatası:\n" + ex.Message + "\n\n" + ex.StackTrace,
                    "VOX Studio",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }
    }
}
