import express from 'express';
import { addMarks } from '../controllers/studentControllers/marks.controllers.js';
import { addStudent } from '../controllers/studentControllers/addStudent.controllers.js';


const router = express.Router();


router.post('/addStudent', addStudent); 
router.post('/addMarks', addMarks);

export default router;