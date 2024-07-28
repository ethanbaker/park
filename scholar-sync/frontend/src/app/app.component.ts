import { Component, ElementRef, Input, OnInit } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { Papa } from 'ngx-papaparse';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
  ],
})
export class AppComponent {
  public copyright: string = "2024";

  selectedFile: File | null = null;

  response: any | null = null;

  constructor(
    private http: HttpClient,
    private papa: Papa,
  ) {

    const currentYear  = new Date().getFullYear();
    if (currentYear != 2024) {
      this.copyright += ` - ${currentYear}`;
    } 
  }

  onFileChange(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length) {
      this.selectedFile = input.files[0];
    }
  }

  onSubmit() {
    if (this.selectedFile) {
      const reader = new FileReader();

      reader.onload = (event: any) => {
        const csvData = event.target.result;
        this.parseCsv(csvData);
      };

      reader.readAsText(this.selectedFile);
    }
  }

  parseCsv(csvData: string) {
    this.papa.parse(csvData, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        this.sendDataToApi(results.data);
      },
      error: (error) => {
        console.error('CSV parsing error:', error);
      }
    });
  }

  sendDataToApi(data: any[]) {
    const apiUrl = 'http://localhost:8000/api/v1/pairs';

    this.http.post(apiUrl, data).subscribe({
      next: (response) => {
        console.log('Data successfully sent to API', response);
        this.response = response;
      },
      error: (error) => {
        console.error('Error sending data to API', error);
      }
    });
  }


}