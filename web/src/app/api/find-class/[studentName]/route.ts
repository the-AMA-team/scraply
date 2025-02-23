import { NextRequest, NextResponse } from "next/server";
import prisma from '../../../../util/connect'

//FIND STUDENT CLASSES

export const GET = async (req: NextRequest, { params }: any) => {
  try {
    const { studentName } = await params;

    const classrooms = await prisma.classrooms.findMany({});

    for (let i = 0; i < classrooms.length; i++) {
      if (classrooms[i].students.includes(studentName)) {
        return NextResponse.json({ classroom: classrooms[i] }, { status: 200 });
      }
    }

    // console.log("Classrooms found:", classrooms);

    // if (!user) {
    //   console.log("User not found");
    //   return null;
    // }

    // console.log("User found:", user);
    return NextResponse.json({ classrooms: classrooms }, { status: 200 });
  } catch (error) {
    console.error("Error retrieving user:", error);
    return NextResponse.json({ error }, { status: 200 });

    throw error;
  }
};
