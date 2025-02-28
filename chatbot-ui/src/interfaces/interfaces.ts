export interface message{
    content:string;
    role:string;
    id:string;
    pdf?:string[];
    sources?:string[];
}