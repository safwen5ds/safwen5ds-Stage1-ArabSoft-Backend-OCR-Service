package main

import (
	"fmt"
	"golang.org/x/sys/windows"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"unsafe"
)

var (
	tempDir string
)

func getDiskWithMostFreeSpace() string {
	drives := "CDEFGHIJKLMNOPQRSTUVWXYZ"
	var maxFreeSpace uint64
	var bestDrive string

	for _, drive := range drives {
		rootPath := fmt.Sprintf("%s:\\", string(drive))
		freeBytesAvailable := uint64(0)
		ret, _, err := windows.NewLazyDLL("kernel32.dll").NewProc("GetDiskFreeSpaceExW").Call(
			uintptr(unsafe.Pointer(windows.StringToUTF16Ptr(rootPath))),
			uintptr(unsafe.Pointer(&freeBytesAvailable)),
			0,
			0,
		)
		if ret != 0 && err == windows.ERROR_SUCCESS && freeBytesAvailable > maxFreeSpace {
			maxFreeSpace = freeBytesAvailable
			bestDrive = rootPath
		}
	}

	if bestDrive == "" {
		log.Fatal("Aucune partition disponible avec suffisamment d'espace.")
	}

	tempDirPath := filepath.Join(bestDrive, "TempFiles")
	if err := os.MkdirAll(tempDirPath, os.ModePerm); err != nil {
		log.Fatalf("Impossible de créer le répertoire temporaire: %v", err)
	}
	return tempDirPath
}

func uploadFile(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Content-Type") != "application/pdf" {
		http.Error(w, "Content-Type n'est pas application/pdf", http.StatusUnsupportedMediaType)
		return
	}

	tempFile, err := os.CreateTemp(tempDir, "FichierTemporaireStage-*.pdf")
	if err != nil {
		log.Printf("Impossible de créer le fichier temporaire: %v", err)
		http.Error(w, "Impossible de créer le fichier temporaire", http.StatusInternalServerError)
		return
	}
	defer func() {
		if err := tempFile.Close(); err != nil {
			log.Printf("Erreur lors de la fermeture du fichier temporaire: %v", err)
		}
		if err := os.Remove(tempFile.Name()); err != nil {
			log.Printf("Erreur lors de la suppression du fichier temporaire: %v", err)
		}
	}()

	_, err = io.Copy(tempFile, r.Body)
	if err != nil {
		log.Printf("Erreur lors de la copie du corps de la requête dans le fichier temporaire: %v", err)
		http.Error(w, "Impossible de copier le contenu de corps de requête dans le fichier temporaire", http.StatusInternalServerError)
		return
	}

	log.Printf("OCR en cours d'execution pour le fichier : %s", tempFile.Name())

	cmd := exec.Command("python", "./Script.py", tempFile.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Erreur lors de l'exécution de l'OCR: %v\nOutput: %s", err, string(output))
		http.Error(w, fmt.Sprintf("Impossible d'exécuter OCR: %v", err), http.StatusInternalServerError)
		return
	}

	w.Write(output)
}

func main() {
	tempDir = getDiskWithMostFreeSpace()

	http.HandleFunc("/OCR", uploadFile)

	server := &http.Server{
		Addr: ":8081",
	}

	log.Println("Serveur OCR démarré ")
	log.Fatal(server.ListenAndServe())
}
